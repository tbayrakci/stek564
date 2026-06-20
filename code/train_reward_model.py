import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from drone_dispatch_env import load_offline_dataset, evaluate, Config

# --- HİPERPARAMETRELER ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
REWARD_EPOCHS = 15  # Ödül modelini eğitmek için
IQL_EPOCHS = 50     # IQL'i yeni ödüllerle eğitmek için
HIDDEN_DIM = 256
EXPECTILE = 0.8
TEMPERATURE = 3.0
GAMMA = 0.99

# --- 1. AĞ MİMARİLERİ ---
class Network(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 2. SİMÜLATÖR POLİTİKASI (181 Boyut Düzeltmesi ile) ---
class IQLPolicy:
    def __init__(self, policy_net):
        self.policy = policy_net
        self.policy.eval()

    def act(self, obs):
        state_features = []
        for key in sorted(obs.keys()):
            if key not in ["action_mask", "grid"]:
                state_features.append(np.array(obs[key]).flatten())
        
        state_vec = np.concatenate(state_features)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.policy(state_tensor)
            mask = obs.get("action_mask")
            if mask is not None:
                logits_np = logits.numpy()[0]
                logits_np = np.where(mask, logits_np, -np.inf)
                action = int(np.argmax(logits_np))
            else:
                action = torch.argmax(logits, dim=1).item()
        return action

# --- 3. ÖDÜL MODELİ (BRADLEY-TERRY) EĞİTİMİ ---
def train_reward_model(states, actions, true_rewards):
    print("\n--- Aşama 1: Tercih Bazlı Ödül Modeli Eğitiliyor ---")
    input_dim = states.shape[1] + 1 # State + Action
    reward_net = Network(input_dim, 1)
    optimizer = optim.Adam(reward_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Eylemleri state ile birleştir
    sa_pairs = torch.cat([states, actions.float()], dim=1)
    
    # Tercih veri seti oluştur (Rastgele ikililer seçip karşılaştırıyoruz)
    # Gerçek ödülü yüksek olana 1, düşük olana 0 etiketi veriyoruz
    n_samples = len(states)
    
    for epoch in range(REWARD_EPOCHS):
        total_loss = 0
        # Basitlik için rastgele indeksler üzerinden eşleştirme yapıyoruz
        idx1 = torch.randint(0, n_samples, (BATCH_SIZE,))
        idx2 = torch.randint(0, n_samples, (BATCH_SIZE,))
        
        sa1, sa2 = sa_pairs[idx1], sa_pairs[idx2]
        r1_true, r2_true = true_rewards[idx1], true_rewards[idx2]
        
        # Eğer r1 > r2 ise tercih 1, değilse 0
        preferences = (r1_true > r2_true).float()
        
        optimizer.zero_grad()
        r1_pred = reward_net(sa1)
        r2_pred = reward_net(sa2)
        
        # Bradley-Terry formülü: Logits farkı
        logits_diff = r1_pred - r2_pred
        loss = criterion(logits_diff, preferences)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        print(f"Ödül Epoch {epoch+1}/{REWARD_EPOCHS} | Loss: {total_loss:.4f}")
        
    return reward_net, sa_pairs

# --- 4. YENİ ÖDÜLLERLE IQL EĞİTİMİ ---
def train_iql_with_recovered_rewards():
    print("Veri yükleniyor...")
    dataset = load_offline_dataset("../data/D_logs.npz")
    
    states = torch.FloatTensor(dataset['observations'])
    actions = torch.LongTensor(dataset['actions']).unsqueeze(1)
    real_rewards = torch.FloatTensor(dataset['rewards']).unsqueeze(1)
    next_states = torch.FloatTensor(dataset['next_observations'])
    dones = torch.FloatTensor(dataset['terminals']).unsqueeze(1)
    
    # 1. Ödül Modelini Eğit
    reward_net, sa_pairs = train_reward_model(states, actions, real_rewards)
    
    # 2. Veri Setindeki Ödülleri "Kurtarılan (Recovered)" Ödüllerle Değiştir
    print("\nOrijinal ödüller siliniyor... Yapay ödüller hesaplanıyor...")
    with torch.no_grad():
        recovered_rewards = reward_net(sa_pairs)
        # Ödülleri standartlaştır (Eğitimi stabilize etmek için)
        recovered_rewards = (recovered_rewards - recovered_rewards.mean()) / (recovered_rewards.std() + 1e-8)
    
    print("\n--- Aşama 2: IQL Modeli Kurtarılan Ödüllerle Eğitiliyor ---")
    input_dim = states.shape[1]
    output_dim = len(torch.unique(actions))
    
    # DİKKAT: Artık 'real_rewards' yerine 'recovered_rewards' kullanıyoruz!
    train_loader = DataLoader(TensorDataset(states, actions, recovered_rewards, next_states, dones), batch_size=BATCH_SIZE, shuffle=True)

    q_net = Network(input_dim, output_dim)
    target_q_net = Network(input_dim, output_dim)
    target_q_net.load_state_dict(q_net.state_dict())
    
    v_net = Network(input_dim)
    policy_net = Network(input_dim, output_dim)
    
    q_opt = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    v_opt = optim.Adam(v_net.parameters(), lr=LEARNING_RATE)
    p_opt = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    for epoch in range(IQL_EPOCHS):
        total_p_loss = 0
        for batch_s, batch_a, batch_r, batch_s_next, batch_done in train_loader:
            # V Ağı (Expectile Loss)
            q_values = target_q_net(batch_s).gather(1, batch_a).detach()
            v_values = v_net(batch_s)
            diff = q_values - v_values
            v_loss = torch.where(diff > 0, EXPECTILE * (diff**2), (1 - EXPECTILE) * (diff**2)).mean()
            v_opt.zero_grad(); v_loss.backward(); v_opt.step()

            # Q Ağı (Yeni Ödüllerle)
            with torch.no_grad():
                next_v = v_net(batch_s_next)
                target_q = batch_r + (GAMMA * next_v * (1 - batch_done))
            current_q = q_net(batch_s).gather(1, batch_a)
            q_loss = nn.MSELoss()(current_q, target_q)
            q_opt.zero_grad(); q_loss.backward(); q_opt.step()

            # Politika Ağı
            with torch.no_grad():
                adv = q_values - v_values
                weight = torch.exp(TEMPERATURE * adv).clamp(max=100.0)
            logits = policy_net(batch_s)
            p_loss = (weight * nn.CrossEntropyLoss(reduction='none')(logits, batch_a.squeeze())).mean()
            p_opt.zero_grad(); p_loss.backward(); p_opt.step()
            total_p_loss += p_loss.item()
            
        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)
            
        if (epoch+1) % 10 == 0:
            print(f"IQL Epoch {epoch+1}/{IQL_EPOCHS} | P Loss: {total_p_loss/len(train_loader):.2f}")

    os.makedirs("../weights", exist_ok=True)
    torch.save(policy_net.state_dict(), "../weights/irl_policy.pt")
    return policy_net

def main():
    trained_policy = train_iql_with_recovered_rewards()
    agent = IQLPolicy(trained_policy)
    
    print("\nSimülatörde Değerlendirme Yapılıyor (Seeds: [0, 1, 2])...")
    eval_config = Config()
    results = evaluate(agent, eval_config, seeds=[0, 1, 2])
    
    if 'mean' in results:
        print(f"Gerçek Ödülle Eğitilen Önceki IQL Maliyeti: 20.32")
        print(f"Kurtarılan (Sahte) Ödülle Eğitilen Yeni Maliyet: {results['mean'].get('cost_per_order', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
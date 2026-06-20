import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from drone_dispatch_env import load_offline_dataset, evaluate, Config

# --- ABLASYON İÇİN HİPERPARAMETRELER ---
# Rapordaki eğriyi çizmek için bu değeri 0.0, 10.0 ve 50.0 yaparak 3 kez çalıştıracağız
LAMBDA_PENALTY = 0.0  

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
HIDDEN_DIM = 256
EXPECTILE = 0.8
TEMPERATURE = 3.0
GAMMA = 0.99

# --- 1. AĞ MİMARİSİ ---
class Network(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Network, self).__init__()
        out_dim = output_dim if output_dim else 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 2. SİMÜLATÖR POLİTİKASI ---
class SafeIQLPolicy:
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

# --- 3. GÜVENLİ IQL (CMDP) EĞİTİM DÖNGÜSÜ ---
def train_safe_iql():
    print(f"Veri yükleniyor... (Lagrangian Ceza Katsayısı [Lambda]: {LAMBDA_PENALTY})")
    dataset = load_offline_dataset("../data/D_logs.npz")
    
    states = torch.FloatTensor(dataset['observations'])
    actions = torch.LongTensor(dataset['actions']).unsqueeze(1)
    rewards = torch.FloatTensor(dataset['rewards']).unsqueeze(1)
    next_states = torch.FloatTensor(dataset['next_observations'])
    dones = torch.FloatTensor(dataset['terminals']).unsqueeze(1)
    
    # GÜVENLİK SİNYALİ (COST) OLUŞTURMA:
    raw_rewards = dataset['rewards']
    costs_np = np.where(raw_rewards < -10, 1.0, 0.0)
    costs = torch.FloatTensor(costs_np).unsqueeze(1)
    
    print(f"Veri setindeki toplam güvenlik ihlali (kaza/batarya bitmesi) sayısı: {int(costs_np.sum())}")
    
    input_dim = states.shape[1]
    output_dim = len(torch.unique(actions))
    
    train_loader = DataLoader(TensorDataset(states, actions, rewards, costs, next_states, dones), batch_size=BATCH_SIZE, shuffle=True)

    q_net = Network(input_dim, output_dim)
    target_q_net = Network(input_dim, output_dim)
    target_q_net.load_state_dict(q_net.state_dict())
    v_net = Network(input_dim)
    policy_net = Network(input_dim, output_dim)
    
    q_optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    v_optimizer = optim.Adam(v_net.parameters(), lr=LEARNING_RATE)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    print("\nGüvenli IQL (Lagrangian) Eğitimi Başlıyor...")

    for epoch in range(EPOCHS):
        total_p_loss = 0
        for batch_s, batch_a, batch_r, batch_c, batch_s_next, batch_done in train_loader:
            
            # V Ağı Eğitimi
            q_values = target_q_net(batch_s).gather(1, batch_a).detach()
            v_values = v_net(batch_s)
            diff = q_values - v_values
            v_loss = torch.where(diff > 0, EXPECTILE * (diff**2), (1 - EXPECTILE) * (diff**2)).mean()
            v_optimizer.zero_grad(); v_loss.backward(); v_optimizer.step()

            # --- LAGRANGIAN MODİFİKASYONU ---
            modified_reward = batch_r - (LAMBDA_PENALTY * batch_c)
            
            # Q Ağı Eğitimi (Ceza yemiş yeni ödülle)
            with torch.no_grad():
                next_v = v_net(batch_s_next)
                target_q = modified_reward + (GAMMA * next_v * (1 - batch_done))
            current_q = q_net(batch_s).gather(1, batch_a)
            q_loss = nn.MSELoss()(current_q, target_q)
            q_optimizer.zero_grad(); q_loss.backward(); q_optimizer.step()

            # Politika Ağı Eğitimi
            with torch.no_grad():
                adv = q_values - v_values
                weight = torch.exp(TEMPERATURE * adv).clamp(max=100.0)
            logits = policy_net(batch_s)
            p_loss = (weight * nn.CrossEntropyLoss(reduction='none')(logits, batch_a.squeeze())).mean()
            policy_optimizer.zero_grad(); p_loss.backward(); policy_optimizer.step()
            total_p_loss += p_loss.item()
            
        # Hedef ağı güncelle
        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)
            
        if (epoch+1) % 10 == 0:
            print(f"CMDP Epoch {epoch+1}/{EPOCHS} | Politika Kaybı: {total_p_loss/len(train_loader):.2f}")

    os.makedirs("../weights", exist_ok=True)
    # Model kaydedilirken lambda değerini ismine ekliyoruz ki testlerde karışmasın
    torch.save(policy_net.state_dict(), f"../weights/cmdp_lambda_{int(LAMBDA_PENALTY)}.pt")
    return policy_net

def main():
    trained_policy = train_safe_iql()
    agent = SafeIQLPolicy(trained_policy)
    
    print("\nSimülatörde Değerlendirme Yapılıyor (Seeds: [0, 1, 2])...")
    eval_config = Config()
    results = evaluate(agent, eval_config, seeds=[0, 1, 2])
    
    if 'mean' in results:
        print(f"\n--- ABLASYON SONUÇLARI (Lambda: {LAMBDA_PENALTY}) ---")
        print(f"Maliyet (cost_per_order) : {results['mean'].get('cost_per_order', 'N/A'):.4f}")
        print(f"Güvenlik İhlalleri (depletion_events) : {results['mean'].get('depletion_events', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
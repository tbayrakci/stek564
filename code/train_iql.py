import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt

from drone_dispatch_env import load_offline_dataset, evaluate, Config

# --- HİPERPARAMETRELER ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
HIDDEN_DIM = 256
GAMMA = 0.99
EXPECTILE = 0.8  
TEMPERATURE = 3.0 

# --- 1. AĞ MİMARİLERİ (IQL, Q ve V ağlarını aynı anda kullanır) ---
class Network(nn.Module):
    def __init__(self, input_dim, output_dim=None):
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

# --- 2. SİMÜLATÖR POLİTİKASI (Aynı Girdi İşleme) ---
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

# --- 3. EĞİTİM DÖNGÜSÜ (IQL) ---
def train_iql():
    print("Veri yükleniyor...")
    dataset = load_offline_dataset("../data/D_logs.npz")
    
    states = torch.FloatTensor(dataset['observations'])
    actions = torch.LongTensor(dataset['actions']).unsqueeze(1)
    rewards = torch.FloatTensor(dataset['rewards']).unsqueeze(1)
    next_states = torch.FloatTensor(dataset['next_observations'])
    dones = torch.FloatTensor(dataset['terminals']).unsqueeze(1)
    
    input_dim = states.shape[1]
    output_dim = len(torch.unique(actions))
    
    train_loader = DataLoader(TensorDataset(states, actions, rewards, next_states, dones), batch_size=BATCH_SIZE, shuffle=True)

    q_net = Network(input_dim, output_dim)
    target_q_net = Network(input_dim, output_dim)
    target_q_net.load_state_dict(q_net.state_dict())
    
    v_net = Network(input_dim) # Değer ağı (Durumun kalitesini ölçer)
    policy_net = Network(input_dim, output_dim) # Aktör ağı
    
    q_optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    v_optimizer = optim.Adam(v_net.parameters(), lr=LEARNING_RATE)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    print(f"\nIQL Eğitimi Başlıyor (Expectile: {EXPECTILE})...")
    average_q_values = []

    for epoch in range(EPOCHS):
        total_q_loss = total_v_loss = total_p_loss = 0
        epoch_q = []
        
        for batch_s, batch_a, batch_r, batch_s_next, batch_done in train_loader:
            # 1. V-Ağı Eğitimi (Expectile Loss)
            q_values = target_q_net(batch_s).gather(1, batch_a).detach()
            v_values = v_net(batch_s)
            diff = q_values - v_values
            # IQL'in büyüsü: Sadece pozitif farkları (iyi eylemleri) ağırlıklandır
            v_loss = torch.where(diff > 0, EXPECTILE * (diff**2), (1 - EXPECTILE) * (diff**2)).mean()
            
            v_optimizer.zero_grad()
            v_loss.backward()
            v_optimizer.step()

            # 2. Q-Ağı Eğitimi
            with torch.no_grad():
                next_v = v_net(batch_s_next)
                target_q = batch_r + (GAMMA * next_v * (1 - batch_done))
                
            current_q = q_net(batch_s).gather(1, batch_a)
            q_loss = nn.MSELoss()(current_q, target_q)
            
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # 3. Politika (Actor) Ağı Eğitimi (Ağırlıklı BC)
            with torch.no_grad():
                adv = q_values - v_values 
                weight = torch.exp(TEMPERATURE * adv).clamp(max=100.0) 
                
            logits = policy_net(batch_s)
            p_loss = (weight * nn.CrossEntropyLoss(reduction='none')(logits, batch_a.squeeze())).mean()
            
            policy_optimizer.zero_grad()
            p_loss.backward()
            policy_optimizer.step()

            total_q_loss += q_loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            epoch_q.append(current_q.detach().mean().item())
            
        # Yumuşak (Soft) hedef ağı güncellemesi
        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)
            
        avg_q = np.mean(epoch_q)
        average_q_values.append(avg_q)
        print(f"Epoch {epoch+1}/{EPOCHS} | Q Loss: {total_q_loss/len(train_loader):.2f} | P Loss: {total_p_loss/len(train_loader):.2f} | Ort. Q: {avg_q:.2f}")

    torch.save(policy_net.state_dict(), "../weights/iql.pt")
    
    # KONTROL GRAFİĞİ
    plt.clf()
    plt.plot(average_q_values, color='blue')
    plt.title("IQL - Stabil Q-Değerleri")
    plt.savefig("../logs/iql_stable_q_values.png")
    
    return policy_net

def main():
    trained_policy = train_iql()
    agent = IQLPolicy(trained_policy)
    
    print("\nSimülatörde Değerlendirme Yapılıyor (Seeds: [0, 1, 2])...")
    eval_config = Config()
    results = evaluate(agent, eval_config, seeds=[0, 1, 2])
    
    if 'mean' in results:
        print(f"Maliyet (cost_per_order): {results['mean'].get('cost_per_order', 'N/A'):.4f}")
        print(f"Başarı Oranı: {results['mean'].get('success_rate', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
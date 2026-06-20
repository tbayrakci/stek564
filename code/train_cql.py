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
CQL_ALPHA = 5.0  

# --- 1. Q-AĞI MİMARİSİ (Aynı) ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. SİMÜLATÖR POLİTİKASI (Aynı) ---
class CQLPolicy:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def act(self, obs):
        state_features = []
        for key in sorted(obs.keys()):
            if key not in ["action_mask", "grid"]:
                state_features.append(np.array(obs[key]).flatten())
        
        state_vec = np.concatenate(state_features)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            mask = obs.get("action_mask")
            if mask is not None:
                q_np = q_values.numpy()[0]
                q_np = np.where(mask, q_np, -np.inf)
                action = int(np.argmax(q_np))
            else:
                action = torch.argmax(q_values, dim=1).item()
                
        return action

# --- 3. EĞİTİM DÖNGÜSÜ (CQL) ---
def train_cql():
    print("Veri yükleniyor...")
    dataset = load_offline_dataset("../data/D_logs.npz")
    
    states = torch.FloatTensor(dataset['observations'])
    actions = torch.LongTensor(dataset['actions']).unsqueeze(1)
    rewards = torch.FloatTensor(dataset['rewards']).unsqueeze(1)
    next_states = torch.FloatTensor(dataset['next_observations'])
    dones = torch.FloatTensor(dataset['terminals']).unsqueeze(1)
    
    input_dim = states.shape[1]
    output_dim = len(torch.unique(actions))
    
    train_data = TensorDataset(states, actions, rewards, next_states, dones)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    q_net = QNetwork(input_dim, output_dim)
    target_net = QNetwork(input_dim, output_dim)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"\nCQL Eğitimi Başlıyor (Alpha: {CQL_ALPHA})...")
    average_max_q_values = []

    for epoch in range(EPOCHS):
        total_bellman_loss = 0
        total_cql_loss = 0
        epoch_q_vals = []
        
        for batch_s, batch_a, batch_r, batch_s_next, batch_done in train_loader:
            # Tüm eylemlerin Q-değerleri
            q_values = q_net(batch_s) 
            # Veri setindeki (uzmanın seçtiği) eylemin Q-değeri
            current_q = q_values.gather(1, batch_a) 
            epoch_q_vals.append(current_q.detach().mean().item())
            
            # --- STANDART DQN KAYBI (Bellman) ---
            with torch.no_grad():
                max_next_q = target_net(batch_s_next).max(1, keepdim=True)[0]
                target_q = batch_r + (GAMMA * max_next_q * (1 - batch_done))
            bellman_loss = criterion(current_q, target_q)
            
            # --- CQL KAYBI (Cezalandırma Mekanizması) ---
            cql_loss = torch.logsumexp(q_values, dim=1, keepdim=True) - current_q
            cql_loss = cql_loss.mean()
            
            loss = bellman_loss + (CQL_ALPHA * cql_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_bellman_loss += bellman_loss.item()
            total_cql_loss += cql_loss.item()
            
        if epoch % 2 == 0:
            target_net.load_state_dict(q_net.state_dict())
            
        avg_q = np.mean(epoch_q_vals)
        average_max_q_values.append(avg_q)
        print(f"Epoch {epoch+1}/{EPOCHS} | Bellman: {total_bellman_loss/len(train_loader):.2f} | CQL Ceza: {total_cql_loss/len(train_loader):.2f} | Ort. Q-Değeri: {avg_q:.2f}")

    torch.save(q_net.state_dict(), "../weights/cql.pt")
    
    plt.clf()
    plt.plot(average_max_q_values, color='green')
    plt.title("CQL - Kontrol Altına Alınmış Q-Değerleri")
    plt.xlabel("Epoch")
    plt.ylabel("Ortalama Tahmini Q-Değeri")
    plt.savefig("../logs/cql_stable_q_values.png")
    
    return q_net

def main():
    trained_model = train_cql()
    agent = CQLPolicy(trained_model)
    
    print("\nSimülatörde Değerlendirme Yapılıyor (Seeds: [0, 1, 2])...")
    eval_config = Config()
    results = evaluate(agent, eval_config, seeds=[0, 1, 2])
    
    if 'mean' in results:
        print(f"Maliyet (cost_per_order): {results['mean'].get('cost_per_order', 'N/A'):.4f}")
        print(f"Başarı Oranı: {results['mean'].get('success_rate', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
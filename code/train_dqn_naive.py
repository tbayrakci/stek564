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
EPOCHS = 20
HIDDEN_DIM = 256
GAMMA = 0.99

# --- 1. Q-AĞI MİMARİSİ ---
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

# --- 2. SİMÜLATÖR POLİTİKASI
class DQNPolicy:
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

# --- 3. EĞİTİM DÖNGÜSÜ (Başarısızlığı Üretme) ---
def train_naive_dqn():
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

    print("\nNaive Offline DQN Eğitimi Başlıyor...")
    
    average_max_q_values = []

    for epoch in range(EPOCHS):
        total_loss = 0
        epoch_q_vals = []
        
        for batch_s, batch_a, batch_r, batch_s_next, batch_done in train_loader:
            # 1. Mevcut Q değerlerini hesapla
            current_q = q_net(batch_s).gather(1, batch_a)
            epoch_q_vals.append(current_q.detach().mean().item())
            
            with torch.no_grad():
                max_next_q = target_net(batch_s_next).max(1, keepdim=True)[0]
                target_q = batch_r + (GAMMA * max_next_q * (1 - batch_done))
            
            loss = criterion(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 2 == 0:
            target_net.load_state_dict(q_net.state_dict())
            
        avg_q = np.mean(epoch_q_vals)
        average_max_q_values.append(avg_q)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.2f} | Ortalama Q-Değeri: {avg_q:.2f}")

    torch.save(q_net.state_dict(), "../weights/naive.pt")
    
    plt.plot(average_max_q_values, color='red')
    plt.title("Naive Offline DQN - Q-Değeri Patlaması (Overestimation)")
    plt.xlabel("Epoch")
    plt.ylabel("Ortalama Tahmini Q-Değeri")
    plt.savefig("../logs/q_value_explosion.png")
    print("\nGrafik 'logs/q_value_explosion.png' olarak kaydedildi.")
    
    return q_net

def main():
    trained_model = train_naive_dqn()
    agent = DQNPolicy(trained_model)
    
    print("\nSimülatörde Değerlendirme Yapılıyor...")
    eval_config = Config()
    results = evaluate(agent, eval_config, seeds=[0, 1, 2])
    
    if 'mean' in results:
        print(f"Maliyet (cost_per_order): {results['mean'].get('cost_per_order', 'N/A'):.4f}")
    else:
        print("Değerlendirme yapılamadı veya çöktü (Beklenen bir sonuç!).")

if __name__ == "__main__":
    os.makedirs("../logs", exist_ok=True)
    main()
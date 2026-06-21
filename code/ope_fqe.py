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
EPOCHS = 15
HIDDEN_DIM = 256
GAMMA = 0.99

# --- 1. AĞ MİMARİSİ
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
            # FQE OPE için logits üzerinden argmax alıyoruz
            action = torch.argmax(logits, dim=1).item()
        return action

# --- 3. FQE (Fitted Q-Evaluation) ALGORİTMASI ---
def run_fqe_ope():
    print("Veri yükleniyor...")
    dataset = load_offline_dataset("../data/D_logs.npz")
    
    states = torch.FloatTensor(dataset['observations'])
    actions = torch.LongTensor(dataset['actions']).unsqueeze(1)
    rewards = torch.FloatTensor(dataset['rewards']).unsqueeze(1)
    next_states = torch.FloatTensor(dataset['next_observations'])
    dones = torch.FloatTensor(dataset['terminals']).unsqueeze(1)
    
    input_dim = states.shape[1]
    output_dim = len(torch.unique(actions))
    
    # EN İYİ MODELİMİZİ YÜKLÜYORUZ (Lambda 10.0 olan CMDP)
    print("\nEn iyi politikamız (cmdp_lambda_10.pt) yükleniyor...")
    target_policy = Network(input_dim, output_dim)
    try:
        target_policy.load_state_dict(torch.load("../weights/cmdp_lambda_10.pt"))
    except FileNotFoundError:
        print("HATA: cmdp_lambda_10.pt bulunamadı! Eğer farklı isimle kaydettiysen burayı düzelt.")
        return
    target_policy.eval()

    # OPE Q-Ağı (Sadece hedef politikanın Q-değerlerini ölçecek)
    q_net = Network(input_dim, output_dim)
    target_q_net = Network(input_dim, output_dim)
    target_q_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_data = TensorDataset(states, actions, rewards, next_states, dones)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    print("FQE (Off-Policy Evaluation) Eğitimi Başlıyor. Bu ağ sadece tahmini puan hesaplayacak...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_s, batch_a, batch_r, batch_s_next, batch_done in train_loader:
            
            with torch.no_grad():
                next_logits = target_policy(batch_s_next)
                next_actions = torch.argmax(next_logits, dim=1).unsqueeze(1)
                
                next_q = target_q_net(batch_s_next).gather(1, next_actions)
                target_q = batch_r + (GAMMA * next_q * (1 - batch_done))
                
            current_q = q_net(batch_s).gather(1, batch_a)
            loss = criterion(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
            target_param.data.copy_(0.01 * param.data + (1.0 - 0.01) * target_param.data)
            
        print(f"FQE Epoch {epoch+1}/{EPOCHS} | Kayıp: {total_loss/len(train_loader):.2f}")

    # --- OPE TAHMİNİ (BÖLÜM GETİRİSİ) ---
    print("\nOPE Tahmini Hesaplanıyor...")
    initial_states = states[:500] 
    with torch.no_grad():
        init_logits = target_policy(initial_states)
        init_actions = torch.argmax(init_logits, dim=1).unsqueeze(1)
        estimated_qs = q_net(initial_states).gather(1, init_actions)
        ope_estimate = estimated_qs.mean().item()
        
    print(f"\n=============================================")
    print(f"SİMÜLATÖRSÜZ (OFFLINE) TAHMİN EDİLEN GETİRİ : {ope_estimate:.2f}")
    print(f"=============================================")
    
    return target_policy

def main():
    target_policy = run_fqe_ope()
    if target_policy is None: return
    
    agent = SafeIQLPolicy(target_policy)
    print("\nGerçek (True) Değer Hesaplanıyor (Simülatör Canlı Testi)...")
    eval_config = Config()
    results = evaluate(agent, eval_config, seeds=[0, 1, 2])
    
    if 'mean' in results:
        print(f"\n=============================================")
        print(f"GERÇEK (TRUE) SİMÜLATÖR GETİRİSİ: {results['mean'].get('episode_return', 'N/A'):.2f}")
        print(f"Maliyet (cost_per_order)        : {results['mean'].get('cost_per_order', 'N/A'):.4f}")
        print(f"=============================================")

if __name__ == "__main__":
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml

from drone_dispatch_env import load_offline_dataset, evaluate, Config

# --- 1. HİPERPARAMETRELER
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 100
HIDDEN_DIM = 128

# --- 2. YAPAY SİNİR AĞI MİMARİSİ (MLP) ---
class BCNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. POLİTİKA ADAPTÖRÜ
class BCPolicy:
    def __init__(self, model):
        self.model = model
        self.model.eval() # Değerlendirme moduna al

    def act(self, obs):
        state_features = []
        for key in sorted(obs.keys()):
            if key not in ["action_mask", "grid"]:
                val_flat = np.array(obs[key]).flatten()
                state_features.append(val_flat)
        
        state_vec = np.concatenate(state_features)
        
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(state_tensor)
            
            mask = obs.get("action_mask")
            if mask is not None:
                logits_np = logits.numpy()[0]
                logits_np = np.where(mask, logits_np, -np.inf)
                action = int(np.argmax(logits_np))
            else:
                action = torch.argmax(logits, dim=1).item()
                
        return action

# --- 4. EĞİTİM DÖNGÜSÜ ---
def train_bc():
    print("Veri yükleniyor...")
    dataset = load_offline_dataset("../data/D_logs.npz")
    
    states = torch.FloatTensor(dataset['observations'])
    actions = torch.LongTensor(dataset['actions'])
    
    input_dim = states.shape[1]
    output_dim = len(torch.unique(actions))
    print(f"Girdi Boyutu: {input_dim}, Çıktı Boyutu (Eylem Sayısı): {output_dim}")

    train_data = TensorDataset(states, actions)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = BCNetwork(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Sınıflandırma problemi için

    print("\nEğitim Başlıyor...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = criterion(predictions, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Ortalama Kayıp (Loss): {total_loss/len(train_loader):.4f}")

    os.makedirs("../weights", exist_ok=True)
    torch.save(model.state_dict(), "../weights/bc.pt")
    print("Model ağırlıkları 'weights/bc.pt' olarak kaydedildi.")
    
    return model

# --- 5. DEĞERLENDİRME ---
def main():
    trained_model = train_bc()
    
    agent = BCPolicy(trained_model)
    
    print("\nSimülatörde Değerlendirme Yapılıyor (Bölüm getirileri hesaplanıyor)...")
    eval_config = Config()
    results = evaluate(agent, eval_config, seeds=[0, 1, 2])
    
    print("\nDeğerlendirme Sonuçları (Seeds: [0, 1, 2]):")
    
    if 'mean' in results:
        for metrik, deger in results['mean'].items():
            print(f"  - {metrik}: {deger:.4f}")
    else:
        print("Simülatörün ham çıktısı:", results)

if __name__ == "__main__":
    main()
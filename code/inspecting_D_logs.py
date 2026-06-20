import numpy as np
import matplotlib.pyplot as plt
from drone_dispatch_env import load_offline_dataset

# 1. Veriyi Yükle
# Hocanın belirttiği veri yolu "data/D_logs.npz" olarak kullanılıyor
print("Veri seti yükleniyor...")
dataset = load_offline_dataset("../data/D_logs.npz") 

# 2. Veri Boyutlarını İncele
print("\n--- Veri Seti İçeriği ---")
print(f"Durum (State) Sayısı: {len(dataset['observations'])}")
print(f"Eylem (Action) Sayısı: {len(dataset['actions'])}")
print(f"Bölüm (Episode) Sayısı: {len(dataset['episode_returns'])}")

# 3. Getiri (Return) Dağılımını Analiz Et
returns = dataset['episode_returns']
print("\n--- Getiri İstatistikleri ---")
print(f"Ortalama Getiri: {np.mean(returns):.2f}")
print(f"Maksimum Getiri: {np.max(returns):.2f}")
print(f"Minimum Getiri: {np.min(returns):.2f}")
print(f"Standart Sapma: {np.std(returns):.2f}")

# Grafikleri Çizdir
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Getiri Dağılımı Grafiği
ax1.hist(returns, bins=50, color='skyblue', edgecolor='black')
ax1.set_title("Bölüm Getirisi (Episode Returns) Dağılımı")
ax1.set_xlabel("Toplam Getiri")
ax1.set_ylabel("Frekans")

# Eylem (Action) Kapsamı Grafiği
actions = dataset['actions']
# Eylemler eğer kesikli (discrete) ise doğrudan saydırabiliriz
unique_actions, counts = np.unique(actions, return_counts=True)
ax2.bar(unique_actions, counts, color='lightgreen', edgecolor='black')
ax2.set_title("Davranış Politikası Eylem Kapsamı")
ax2.set_xlabel("Eylem ID")
ax2.set_ylabel("Seçilme Sayısı")
ax2.set_xticks(unique_actions)

plt.tight_layout()

plt.savefig("veri_kesfi.png", dpi=300, bbox_inches='tight')
print("Grafik 'veri_kesfi.png' ismiyle kaydedildi.")
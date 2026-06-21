import argparse
import torch
import os

from drone_dispatch_env import evaluate, Config

from train_BC import BCNetwork, BCPolicy
from train_dqn_naive import QNetwork as DQNNet, DQNPolicy
from train_cql import QNetwork as CQLNet, CQLPolicy
from train_iql import Network as IQLNet, IQLPolicy
from train_cmdp_lagrangian import Network as CMDPNet, SafeIQLPolicy

INPUT_DIM = 181
OUTPUT_DIM = 169

def evaluate_model(model_name, policy_class, net_class, weight_path, seeds, eval_config):
    """Belirtilen modeli yükler ve simülatörde test eder."""
    print(f"\n[{model_name}] Yükleniyor ve Değerlendiriliyor...")
    
    if not os.path.exists(weight_path):
        print(f"  -> HATA: '{weight_path}' bulunamadı. Lütfen modelin eğitildiğinden emin olun.")
        return

    try:
        model = net_class(INPUT_DIM, OUTPUT_DIM)
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        
        # Ajanı (Policy) oluştur
        agent = policy_class(model)
        
        # Simülatörde test et
        results = evaluate(agent, eval_config, seeds=seeds)
        
        if 'mean' in results:
            cost = results['mean'].get('cost_per_order', 'N/A')
            success = results['mean'].get('success_rate', 'N/A')
            depletion = results['mean'].get('depletion_events', 'N/A')
            print(f"  -> Maliyet (cost_per_order): {cost:.4f}")
            print(f"  -> Başarı Oranı: {success:.4f}")
            print(f"  -> Batarya Bitişi (Depletion): {depletion:.4f}")
        else:
            print("  -> Değerlendirme başarısız (Model muhtemelen çöktü - DQN için beklenir).")
            
    except Exception as e:
        print(f"  -> BEKLENMEYEN HATA: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tüm çevrimdışı RL modellerini test et")
    parser.add_argument("--config", type=str, default="../configs/eval_standard.yaml", help="Değerlendirme config dosyasının yolu")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Virgülle ayrılmış test tohumları (seed)")
    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    
    eval_config = Config() 
    
    print("="*60)
    print(f"STEK 564 TERM PROJECT - OFFLINE RL EVALUATION PIPELINE")
    print(f"Test Tohumları (Seeds): {seed_list}")
    print("="*60)

    # 1. Behavioral Cloning (Baseline)
    evaluate_model("1. BC Baseline", BCPolicy, BCNetwork, "../weights/bc.pt", seed_list, eval_config)
    
    # 2. Naive Offline DQN (Çöken Model)
    evaluate_model("2. Naive Offline DQN", DQNPolicy, DQNNet, "../weights/naive.pt", seed_list, eval_config)
    
    # 3. Conservative Q-Learning (CQL)
    evaluate_model("3. Conservative Q-Learning (CQL)", CQLPolicy, CQLNet, "../weights/cql.pt", seed_list, eval_config)
    
    # 4. Implicit Q-Learning (IQL)
    evaluate_model("4. Implicit Q-Learning (IQL)", IQLPolicy, IQLNet, "../weights/iql.pt", seed_list, eval_config)
    
    # 5. Inverse RL (Tercih Bazlı Ödül)
    evaluate_model("5. IRL Preference Policy", IQLPolicy, IQLNet, "../weights/irl_policy.pt", seed_list, eval_config)
    
    # 6. CMDP / Safe RL (Ablasyonun Tatlı Noktası)
    evaluate_model("6. Safe IQL (CMDP - Lambda 10.0)", SafeIQLPolicy, CMDPNet, "../weights/cmdp_lambda_10.pt", seed_list, eval_config)
    
    print("\n" + "="*60)
    print("Tüm değerlendirmeler tamamlandı.")
    print("="*60)
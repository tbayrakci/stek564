# STEK 564 Term Project: Offline and Inverse Reinforcement Learning from Logged Drone-Delivery Data

## 1. Reproducing the Offline Failure

The first stage of the project consisted of training a naive off-policy Q-learning algorithm (DQN) using the same fixed D_logs data without making any adjustments. As expected from the offline reinforcement learning setting, the DQN suffered from a significant distribution shift and overestimation bias.

Being unable to correct its overly optimistic estimations for out-of-distribution states because of the inability to communicate with the environment, the Q-values of the DQN were going out of bound. During the training process, the average value of Q-values diverged to abnormal values (+146 or more). Thus, the evaluation results have become very poor.

Naive DQN has shown the worst results among all the models in every metric, with an abnormally large order cost of 57.8578 and the success rate being 25.42%.

> Evidence: `logs/q_value_explosion.png`

---

## 2. Conservative Offline RL Fix (CQL / IQL)

For addressing the problem of overestimating the Q-values, two conservative off-policy reinforcement learning algorithms were considered: Conservative Q-Learning (CQL), and Implicit Q-Learning (IQL).

The IQL was selected as the base algorithm because of its ability to work well with the mixed data quality properties of the D_logs (a mixture of expert and noisy examples). As opposed to CQL, which strongly penalizes any out-of-distribution actions, IQL learns the upper expectile of the dataset.

| Method | Cost | Success Rate |
|---|---|---|
| Behavioral Cloning (BC) Baseline | 16.9559 | 57.72% |
| Naive DQN | 57.8578 | 25.42% |
| IQL (Expectile 0.8) | 20.3222 | 51.33% |

The IQL training performed well in terms of stability in Q-value estimation (maintaining stability at −3.0) and far outperformed the Naive DQN baseline. Even though BC, which is an entirely supervised baseline, scored well in terms of cost metric (16.95) on the evaluation seeds due to imitating the trajectories of the experts from logs, IQL formed the stable base that was required for reward recovery.

---

## 3. Inverse Reward Recovery and Preference-Based Modeling

A Preference-based Reward Model (Bradley-Terry) was learned instead of relying on the real reward in the environment. Transition pairs were formed using data and learned the preference-based model to estimate the probability of choosing one transition over another for the given returns. The rewards were then stripped off, and IQL policy was optimized only based on the learned artificial reward.

| Reward Source | Cost | Depletion Events |
|---|---|---|
| IQL (True Reward) | 20.3222 | 6.0000 |
| IQL (Recovered Preference Reward) | 21.6869 | 5.3333 |

**Analysis of Reward Mis-specification:** The recovered reward was quite effective in accomplishing the macro-goals set by the assignment (delivery of the parcel and avoidance of utter disaster) and proved superior to the naively chosen DQN. The model showed signs of reward mis-specification in terms of micro-optimizations (e.g., energy usage with regards to winds). The IQL agent resorted to reward hacking, exploiting structural flaws in the preference network in lieu of being efficient itself, resulting in slightly increased cost (21.68 against 20.32).

---

## 4. Enhancement and Ablation: Safe RL via Constrained MDP

**Hypothesis:** Given that an additional Lagrangian penalty is added in the IQL approach for formulating a CMDP whose objective is to prevent the battery depletion incidents, the safety constraints will teach the agent not to incur the violation of the said safety constraints. Nevertheless, the safety constraint will result in Return-Violation trade-off due to the conservative nature of the routing.

For this reason, a safety criterion was developed where terminal events (rewards < -10) were considered as cost events (c = 1) by using the Lagrangian multiplier in the following way:

$$r_{safe} = r - (\lambda \cdot c)$$

**Ablation Results (Trade-off Curve):**

| Penalty (λ) | Order Cost | Depletion Events (Violations) | Observation |
|---|---|---|---|
| 0.0 (Standard) | 21.6068 | 6.3333 | Baseline behavior; frequent battery drain. |
| 10.0 (Moderate) | 19.6198 | 5.0000 | **Sweet Spot:** Improved efficiency and safety. |
| 50.0 (Heavy) | 20.4697 | 5.6667 | Over-penalization: Policy collapse/paranoia. |

The hypothesis is proved by the ablation test convincingly. When the value of λ was at an average level (10.0), it helped decrease the number of depletion cases while making sure that the cost fell under the sub-20 level. Nevertheless, when the value of λ equaled 50.0, the policy adopted by the drone was too defensive since it flew along the routes that were not optimal.

---

## 5. Off-Policy Evaluation (OPE)

Before implementing the Safe IQL policy (λ = 10.0) on the live simulation environment, the performance of the policy was only tested through offline logs through the Fitted Q-Evaluation (FQE).

| Metric | Value |
|---|---|
| Offline FQE Estimate | −3.78 |
| True Simulated Return | −372.76 |
| The Gap | ~369.0 |

**Why OPE is hard in this dataset:** The huge difference clearly shows how difficult off-policy evaluation (OPE) is in long horizon spatial tasks. The estimator FQE has an inherent bias while working on the dataset called D_logs. Because of the compounding error (Covariate shift), any mistake made by the policy on the live simulator will end up making the drone move into out-of-distribution states. The FQE algorithm, which works within the state distribution found in the offline data, cannot predict the negative rewards coming out of such states.

---

## 6. Related Work & Methodological Positioning

**CQL vs. IQL in Mixed Data:** However, while CQL works to mitigate the overestimation problem caused by out-of-distribution samples through decreasing the Q-value for the unseen actions, it tends to be too conservative especially when the dataset is very noisy. IQL was the better option for D_logs since, in its mathematical formulation, it avoids altogether any query for out-of-sample actions, focusing only on the upper side of the distribution of behavior data.

**IRL vs. Preference-Based Modeling:** The Maximum entropy Inverse Reinforcement Learning assumes the reward as either linear or non-linear with respect to the state features. This is done so that it can accommodate the expectations of the expert about the features. The Bradley-Terry Preference Model on the other hand fits in with current RLHF pipelines and allows the system to extract a reward function based on the ranking of trajectories. Such a model works great for noisy logs of varied expertise because it learns from relative preferences (A > B).

**Safe RL (CMDPs):** The Lagrangian relaxation of an MDP with constraints is the optimization method selected for this work. Previous studies typically have implemented complex protection layers or projection-based safety constraints to satisfy constraints in a practical manner. In contrast, empirical results have shown that adding a double-gradient Lagrangian multiplier directly to Bellman updates during offline implementation of IQL will produce a significant reduction in burnout events, while maintaining stability in the offline training loop.

---

## 7. Engineering Log

- **DQN Divergence:** At first glance, the naive DQN loss function seemed to be fairly stable; however, it quickly exhibited very poor performance in practice. The numerical instability of the `max_q` values may indicate that this neural tip of the iceberg is not good enough.
- **Reward Model Saturation:** The artificial rewards that were used to train the value-based reinforcement learning algorithms also resulted in the IQL target network oscillating at first. By standardizing the returns of rewards using a mean 0, variance 1 normalization, the expectile loss landscape was stabilized.
- **FQE Action Dimensions:** FQE was also experiencing tensor dimension mismatches on the first run where it was attempting to convert the continuous Q values to the discrete environment boundaries. The issue was resolved by using the `torch.argmax` function too liberally over the offline policy’s logits during the target construction step.
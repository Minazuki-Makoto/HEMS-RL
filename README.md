
## 简介
HEMS-RL 是一个基于深度强化学习（Deep Reinforcement Learning, DRL）的家庭能源管理系统（Home Energy Management System, HEMS）仿真与优化平台。本项目实现了三种先进的DRL算法——SAC (Soft Actor-Critic)、DDPG (Deep Deterministic Policy Gradient) 和 PPO (Proximal Policy Optimization)，用于解决家庭用电设备的实时调度优化问题，目标是在满足用户舒适度的前提下，最小化电费支出并平衡电网负荷。

项目提供了一个模块化的仿真环境，支持自定义电价模式、光伏发电、储能系统（BESS）以及多种可控负载（如空调、电动车、洗碗机等）。通过对比不同算法的性能，帮助研究者快速验证和部署RL-based HEMS策略。
本项目实现了三种先进的 **DRL 算法**：

- **SAC (Soft Actor-Critic)**  
- **DDPG (Deep Deterministic Policy Gradient)**  
- **PPO (Proximal Policy Optimization)**

## 算法原理

### 1. SAC（Soft Actor-Critic）

SAC 是一种基于最大熵强化学习的 off-policy 算法，核心思想是：

- 最大化期望奖励，同时最大化策略熵
- 使用双 Q 网络和策略网络
- 提高探索性和收敛稳定性

适用于连续动作空间，尤其适合 HEMS 中储能和负荷调控的连续控制问题。

### 2. DDPG（Deep Deterministic Policy Gradient）

DDPG 是一种基于 Actor-Critic 的深度强化学习算法：

- Actor 输出确定性动作
- Critic 评估动作的 Q 值
- 使用经验回放和目标网络提高稳定性

适合高维连续动作空间，但探索能力相对 SAC 弱。

### 3. PPO（Proximal Policy Optimization）

PPO 是一种 on-policy 算法，核心特点：

- 使用剪切策略更新（clipped objective）避免策略更新过大
- 可以处理连续和离散动作空间
- 收敛稳定性高，适合实验对比基准

---
## 环境配置

建议使用 **Python 3.9+**，推荐创建虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境 (Windows)
venv\Scripts\activate
# 激活虚拟环境 (Linux/Mac)
source venv/bin/activate
```
## 结果展示：
## 1.训练结果展示
<img width="889" height="797" alt="屏幕截图 2026-03-12 013329" src="https://github.com/user-attachments/assets/63f953a9-6a09-480d-a52c-3fa0a00ece7a" />
# 数据结果：
<img width="607" height="137" alt="屏幕截图 2026-03-12 012022" src="https://github.com/user-attachments/assets/043b7b34-1723-4c4a-a14b-527c0973c629" />

## 2.总负荷功率：
<img width="1721" height="1007" alt="屏幕截图 2026-03-12 013338" src="https://github.com/user-attachments/assets/99f38f6c-f6e4-434f-ae5a-d0d8c7158599" />

## 3.冰箱，洗衣机，洗碗机负荷：
<img width="1207" height="875" alt="屏幕截图 2026-03-12 013343" src="https://github.com/user-attachments/assets/fa0cdd73-0c47-4bcb-920d-cefa68013d52" />

## 4.电灯，加湿器，空调负荷：
<img width="1174" height="858" alt="屏幕截图 2026-03-12 013348" src="https://github.com/user-attachments/assets/b22a6a41-1ef3-43be-9161-6f53455ee2a8" />

## 5.电动汽车，电池：
<img width="1696" height="985" alt="屏幕截图 2026-03-12 013354" src="https://github.com/user-attachments/assets/cbde2896-f448-4d70-8535-493045e91db7" />

## 6.PV（光伏发电）:
<img width="1177" height="855" alt="屏幕截图 2026-03-12 013358" src="https://github.com/user-attachments/assets/cce550e6-8f53-47e5-a979-9c5a1eb6423e" />

## 7.电动汽车剩余电量：
<img width="1199" height="859" alt="屏幕截图 2026-03-12 013403" src="https://github.com/user-attachments/assets/fd02fc29-8863-4286-a5fa-288c5ea6bbd3" />



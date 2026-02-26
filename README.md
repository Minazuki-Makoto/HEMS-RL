
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
## 结果展示：<img width="1158" height="1001" alt="屏幕截图 2026-02-25 235407" src="https://github.com/user-attachments/assets/a3b1250f-ae21-43c3-8f54-72db3281aeb9" />


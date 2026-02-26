from PPO import PPO_Agent
from Env import env
from data import price_t,T_t
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SAC import SAC_Agent
import random
import torch
from DDPG import ddpg_agent


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_curve(data, window=20):
    data = np.array(data)
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed


if __name__ == '__main__':
    t=12
    T_primary=T_t(12)-4
    trans_MW_t_start=15
    trans_MW_t_end=22
    trans_alltime=2
    trans_WM_P_set=0.4
    trans_load_ws=0.5
    trans_DIS_t_start=19
    trans_DIS_t_end=22
    trans_DIS_P_set=0.5
    con_load_light_ws=2.0
    con_load_light_min=0.3
    con_load_light_set=0.5
    con_load_Humid_ws=1.1
    con_load_Humid_min=0.2
    con_load_Humid_set=0.3
    HVAC_p_set=2.0
    T_best=24
    HVAC_ws=0.05
    alpha=0.05
    beta=0.85
    error=1.5
    loss=0.03
    P_set=7.5
    energy_eta=0.95
    t_get=8
    t_leave=22
    SOC=60
    SOC_primary=40
    anxiety=0.05
    damage=0.01
    punish=0.5
    ESS_P_set=10
    SOC_max=30
    SOC_min=15
    SOC_initial=21
    energy_convert=0.95
    PV_P_set=4
    state_dim=8
    hidden_dim=128
    action_dim=8
    eps=0.1
    gamma=0.96
    tau=0.005

    a=0.2
    batch_size=40
    buffer_size=10000
    history_rewards=[]

    noise_std=0.0025
    SEED=40
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    environment=env(t,price_t,T_t,T_primary,
    trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
    con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
    HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss,
    P_set,energy_eta,t_get,t_leave,SOC,SOC_primary,anxiety,damage,punish,
    ESS_P_set,SOC_max,SOC_min,SOC_initial,energy_convert,PV_P_set)

    environmentsac=env(t,price_t,T_t,T_primary,
    trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
    con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
    HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss,
    P_set,energy_eta,t_get,t_leave,SOC,SOC_primary,anxiety,damage,punish,
    ESS_P_set,SOC_max,SOC_min,SOC_initial,energy_convert,PV_P_set)

    environmentddpg=env(t,price_t,T_t,T_primary,
    trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
    con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
    HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss,
    P_set,energy_eta,t_get,t_leave,SOC,SOC_primary,anxiety,damage,punish,
    ESS_P_set,SOC_max,SOC_min,SOC_initial,energy_convert,PV_P_set)

    agent=PPO_Agent(state_dim,hidden_dim,action_dim,eps,gamma)
    sac_agent=SAC_Agent(state_dim,hidden_dim,action_dim,gamma,tau,a,buffer_size,batch_size)
    Ddpg_agent=ddpg_agent(state_dim,hidden_dim,action_dim,gamma,noise_std,tau,buffer_size,batch_size)

    action_history=[]
    action_sac_history=[]
    history_rewards_sac=[]
    history_rewards_ddpg=[]
    for alt in range(16000):
        state_primary=environment.reset()
        state_sac_primary=environmentsac.reset()
        state_ddpg_primary=environmentddpg.reset()
        states=[]
        action_all=[]
        action_all_sac=[]
        reward_all=[]
        next_state_all=[]
        dones=[]
        prob_sums=[]
        rewards=0
        rewards_sac=0
        rewards_ddpg=0
        while 1:
            action,log_prob_sum=agent.choose_action(state_primary)
            action_sac,log_probsac=sac_agent.choose_action(state_sac_primary)
            action_np = action
            next_state_sac,reward_sac,done_sac=environmentsac.step(action_sac)
            next_state,reward,done=environment.step(action_np)
            states.append(state_primary)
            action_all.append(action_np)
            reward_all.append(reward)
            next_state_all.append(next_state)
            dones.append(done)
            prob_sums.append(log_prob_sum)
            rewards+=reward
            rewards_sac+=reward_sac
            sac_agent.buffer.push(state_sac_primary,action_sac,reward_sac,next_state_sac,done_sac)

            action_ddpg=Ddpg_agent.choose_action(state_ddpg_primary)
            next_state_ddpg,reward_ddpg,done_ddpg=environmentddpg.step(action_ddpg)
            Ddpg_agent.buffer.push(state_ddpg_primary,action_ddpg,reward_ddpg,next_state_ddpg,done_ddpg)
            rewards_ddpg+=reward_ddpg
            action_all_sac.append(action_sac)
            if done :
                break
            state_primary=next_state
            state_sac_primary=next_state_sac
            state_ddpg_primary=next_state_ddpg

            sac_agent.update()
            sac_agent.soft_update()

        action_history.append(action_all.copy())
        history_rewards .append(rewards)
        history_rewards_sac.append(rewards_sac)
        history_rewards_ddpg.append(rewards_ddpg)
        action_sac_history.append(action_all_sac)
        Ddpg_agent.update()
        agent.update(states, action_all, reward_all, next_state_all, dones, prob_sums)

        if alt % 1000 == 0 :
            print(f'第{alt}次迭代完成，PPO得到的回报为{rewards*50}')
            print(f'SAC算法得到的回报为{rewards_sac*50}')
            print(f'DDPG算法得到的回报为{rewards_ddpg*50}')


        action_all.clear()
        states.clear()
        next_state_all.clear()
        dones.clear()
        prob_sums.clear()
        reward_all.clear()

    history_rewards=[history_rewards[i]*50 for i in range(len(history_rewards))]
    history_rewards_sac=[history_rewards_sac[i]*50 for i in range(len(history_rewards_sac))]
    history_rewards_ddpg=[history_rewards_ddpg[i]*50 for i in range(len(history_rewards_ddpg))]
    alt=[i for i in range(len(history_rewards))]
    smooth_rewards = smooth_curve(history_rewards, window=50)
    smooth_rewards_sac=smooth_curve(history_rewards_sac,window=50)
    smooth_rewards_ddpg=smooth_curve(history_rewards_ddpg,window=50)
    smooth_alt = alt[:len(smooth_rewards)]
    best_idx=np.argmax(history_rewards)
    best_action=action_history[best_idx]
    best_idx_sac=np.argmax(history_rewards_sac)
    ele=['冰箱','MW','DIS','LW','HUM','HVAC','EV','ESS','PV']
    P=[trans_WM_P_set,trans_DIS_P_set,con_load_light_set,con_load_Humid_set,HVAC_p_set,PV_P_set,ESS_P_set,PV_P_set]

    P_fridge=[]
    P_MW=[]
    P_DIS=[]
    P_Light=[]
    P_Humid=[]
    P_HVAC=[]
    P_EV=[]
    P_ESS=[]
    P_PV=[]
    MW_remaintime=[]
    DIS_remaintime=[]
    EV_remain_SOC=[]
    ESS_remain_SOC=[]

    '''写入我的最优策略'''
    print(f'PPO最大价值回报为{history_rewards[best_idx]}')
    print(f'SAC最大价值回报为{history_rewards_sac[best_idx_sac]}')
    print(f'DDPG最大价值回报为{max(history_rewards_ddpg)}')
    best_action=action_sac_history[best_idx_sac]
    environmentsac.reset()
    for i in range(len(best_action)):
        t= 12 + i
        #分配动作
        MW_action=best_action[i][0]
        DIS_action=best_action[i][1]
        Light_action=best_action[i][2]
        Humid_action=best_action[i][3]
        HVAC_action=best_action[i][4]
        EV_action=best_action[i][5]
        ESS_action=best_action[i][6]
        PV_action=best_action[i][7]
        #负荷
        p_fridge = environmentsac.set_load()
        _,p_MW,MW_REMAIN = environmentsac.transform_MW_load(t,MW_action)
        _,p_DIS,DIS_REMAIN= environmentsac.DISWH(t,DIS_action)
        _,p_Light=environmentsac.controlable_load_Light(t,Light_action)
        _,p_Humid=environmentsac.controlable_load_Humid(t,Humid_action)
        _,p_HVAC,T_in=environmentsac.HVAC(t,HVAC_action)
        _,p_EV,_,EV_REMAIN_SOC = environmentsac.EV(t,EV_action)
        _,p_ESS,ESS_REMAIN_SOC = environmentsac.ESS(t,ESS_action)
        p_PV= environmentsac.PV(t,PV_action)

        P_fridge.append(p_fridge)
        P_MW.append(p_MW)
        P_DIS.append(p_DIS)
        P_Light.append(p_Light)
        P_Humid.append(p_Humid)
        P_HVAC.append(p_HVAC)
        P_EV.append(p_EV)
        P_ESS.append(p_ESS)
        P_PV.append(p_PV)
        MW_remaintime.append(MW_REMAIN)
        DIS_remaintime.append(DIS_REMAIN)
        EV_remain_SOC.append(EV_REMAIN_SOC)
        ESS_remain_SOC.append(ESS_REMAIN_SOC)
    time_list = [(12 + i)%24 for i in range(len(P_fridge))]

    data = {
        "time": time_list,
        "P_fridge": P_fridge,
        "P_MW": P_MW,
        "P_DIS": P_DIS,
        "P_Light": P_Light,
        "P_Humid": P_Humid,
        "P_HVAC": P_HVAC,
        "P_EV": P_EV,
        "P_ESS": P_ESS,
        "P_PV": P_PV,
        "MW_remaintime": MW_remaintime,
        "DIS_remaintime": DIS_remaintime,
        "EV_remain_SOC": EV_remain_SOC,
        "ESS_remain_SOC": ESS_remain_SOC
    }

    df = pd.DataFrame(data)
    df.to_csv("D:/pycharmcode/project/HEMS/optimal_strategy.csv", index=False, encoding="utf-8-sig")


    plt.figure(figsize=(18, 16))
    plt.grid(True)
    plt.title('不同算法下的HEMS问题')
    plt.xlabel('迭代次数')
    plt.ylabel('价值回报')
    plt.xlim(0, len(history_rewards) + 1)
    plt.plot(alt, history_rewards, color='lightgreen', lw=1, alpha=0.35, label='多步PPO原始回报')
    plt.plot(smooth_alt, smooth_rewards, color='green', lw=2, label='多步PPO平滑回报')
    plt.plot(alt,history_rewards_sac,color='lightblue',lw=1,alpha=0.35, label='SAC原始回报')
    plt.plot(smooth_alt,smooth_rewards_sac,color='blue',lw=2,label='SAC平滑回报')
    plt.plot(alt, history_rewards_ddpg, color='lightcoral', lw=1, alpha=0.35, label='DDPG原始回报')
    plt.plot(smooth_alt, smooth_rewards_ddpg, color='red', lw=2, label='DDPG平滑回报')
    plt.legend()
    plt.show()



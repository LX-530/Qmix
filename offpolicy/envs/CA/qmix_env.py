import numpy as np
import gym
from gym import spaces
import pandas as pd
import os
from datetime import datetime
from offpolicy.envs.CA.robot_env import RobotEnvironment

class Robot_Env(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        
        target_area = (3, 32, 2, 7) 
        num_persons = 150 
        max_steps = 150
        map_path = r"F:\1Business_code\20250910-qmix_fire\qmix\offpolicy\envs\CA\map.json"
        
        self.robot_env = RobotEnvironment(map_path, target_area, num_persons, max_steps)
        self.robot_env.set_robot_repulsion_factor(0.7)  # 设置为火源斥力的70%

        self.episode_reward = 0

        now = datetime.now()
        self.out_res = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.out_csv_path = "../results/" + self.out_res

        self.agent_num = 2
        self.obs_dim = 10
        self.action_dim = 5
        self.episode_reward, self.epiosde_person_escaped, self.epiosde_person_dead, self.episode_heath = 0, 0, 0, 0
        self.render = False  # 是否可视化
        self.eval = False  # 是否为评估模式
        if self.eval:
            self.eval_episode = 0
            self.eval_csv_path = self.out_csv_path
            # 创建评估输出目录本身，而不是父目录
            os.makedirs(self.eval_csv_path, exist_ok=True)
            print(f"创建评估模式的CSV文件夹{self.eval_csv_path}")

        # 确保输出目录存在
        # os.makedirs(os.path.dirname(self.out_csv_path), exist_ok=True)
        
        # 初始化用于保存每个时间步信息的列表
        self.step_infos = []
        self.total_infos = []

    def reset(self):
        """
        # 重启环境，并返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        s = self.robot_env._reset_environment()
        state = []
        for key, value in s.items():
            state.append(value)
        self.episode_reward, self.epiosde_person_escaped, self.epiosde_person_dead, self.episode_heath = 0, 0, 0, 0
        
        # 重置每个episode的步骤信息记录
        self.step_infos = []

        return state

    def step(self, actions):  # 定义obs、reward的更新方式，这里是随机生成的，可根据自己的项目对更新方式进行定义
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """

        act = [actions[0][0], actions[1][0]]
        state, rewards, done, info = self.robot_env.step(act)
        if self.render:
            self.robot_env.render()

        next_state = list(state.values())
        all_reward = np.stack([[value] for value in rewards])
        all_done = np.stack([[done] for i in range(self.agent_num)])
        self.episode_reward += sum(rewards)
        self.episode_heath += info["avg_health"]
        
        # 如果是eval模式，保存每个时间步的信息
        if self.eval:
            step_info = {
                "episode": self.eval_episode,
                "step": info["step"],
                "persons_remaining": info["persons_remaining"],
                "persons_escaped": info["persons_escaped"],
                "persons_dead": info["persons_dead"],
                "avg_health": info["avg_health"],
                "robot1_reward": rewards[0],
                "robot2_reward": rewards[1],
                "done": done
            }
            self.step_infos.append(step_info)



        # 统计信息每回合的评价指标并保存至Excel
        if done:
            episode_info = {
                "mean_reward": self.episode_reward / self.agent_num,
                "person_escaped": info["persons_escaped"],
                "person_dead": info["persons_dead"],
                "time-consuming": info["step"],
                "avg_health": self.episode_heath / info["step"]
            }
            self.total_infos.append(episode_info)

            print("Done ", self.episode_reward, self.episode_reward / self.agent_num)
            
            if not self.eval:
                # 训练模式保存总体信息
                os.makedirs(self.out_csv_path, exist_ok=True)
                df = pd.DataFrame(self.total_infos)
                df.to_csv(self.out_csv_path + "/qmix_robot.csv", index=False)
            else:
                # eval模式保存每个episode的详细步骤信息
                if self.step_infos:
                    # 确保评估输出目录存在
                    os.makedirs(self.eval_csv_path, exist_ok=True)
                    # 为每个episode创建不同的CSV文件名
                    eval_save_path = self.out_csv_path + f"/episode_{self.eval_episode}.csv"
                    df_steps = pd.DataFrame(self.step_infos)
                    df_steps.to_csv(eval_save_path, index=False)
                    print(f"保存了episode_{self.eval_episode}_steps.csv")

                self.eval_episode += 1


        return next_state, all_reward, all_done, info


if __name__ == '__main__':
    i = 0
    env = Robot_Env(i)

    env.reset()

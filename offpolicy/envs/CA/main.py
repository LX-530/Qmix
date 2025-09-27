# main.py
from operator import le
import numpy as np
import matplotlib.pyplot as plt
from map_loader import MapLoader
from person_behavior import Person
from fire_model import FireModel
from robot_env import RobotEnvironment
import random

def visualize(map_data, persons, fires, robots, step):
    vis_map = np.copy(map_data)
    for person in persons:
        x, y = person.position
        if person.is_dead:
            vis_map[x, y] = 6  # 死亡人员标记
        else:
            vis_map[x, y] = 4  # 活着的人员标记
    for fx, fy in fires:
        vis_map[fx, fy] = 3  # Fire
    for rx, ry in robots:
        vis_map[rx, ry] = 5  # Robot marker

    plt.imshow(vis_map, cmap='hot', interpolation='nearest')
    plt.title(f'Step {step}')
    plt.show(block=False)
    plt.pause(0.05)
    plt.clf()

# Main simulation using Robot Environment
target_area = (3, 32, 2, 7) 
num_persons = 150 
max_steps = 300

# 初始化机器人环境
robot_env = RobotEnvironment('map.json', target_area, num_persons, max_steps)

# 可选：调节机器人斥力因子
robot_env.set_robot_repulsion_factor(0.7)  # 设置为火源斥力的30%

print(f"初始化环境: {robot_env.initial_person_count} 名人员需要疏散")
print(f"机器人斥力因子: {robot_env.robot_repulsion_factor}")
print(f"机器人1(火源防护)位置: {robot_env.robot_positions[0] if len(robot_env.robot_positions) > 0 else '未部署'}")
print(f"机器人2(出口疏散)位置: {robot_env.robot_positions[1] if len(robot_env.robot_positions) > 1 else '未部署'}")
print(f"火源位置: {list(robot_env.map_loader.fires)}")
print(f"出口位置: {list(robot_env.map_loader.exits)}")

# Simulation loop
for step in range(max_steps):
    # 随机生成两个机器人的动作 (0:上, 1:下, 2:左, 3:右, 4:静止)
    robot_actions = [random.randint(0, 4), random.randint(0, 4)]
    # robot_actions = [4, 4]

    
    # 执行一步仿真
    state, reward, done, info = robot_env.step(robot_actions)
    print(state, "\n", len(state))
    print(reward)
    
    
    # 可视化
    visualize(robot_env.map_loader.map_data, robot_env.persons, 
              robot_env.map_loader.fires, robot_env.robot_positions, step)
    
    # 打印信息
    if step % 20 == 0:  # 每20步打印一次信息
        breakdown = robot_env.reward_breakdown
        print(f"步骤 {step}: 存活 {info['persons_remaining']} 人, "
              f"已疏散 {info['persons_escaped']} 人, "
              f"死亡 {info['persons_dead']} 人")
        print(f"  共享奖励 - 撤离:{breakdown['shared_evacuation']}, 健康:{breakdown['shared_health']}, "
              f"协作:{breakdown['shared_cooperation']}, 完成:{breakdown['shared_completion']}")
        print(f"  个体奖励 - 机器人1:{breakdown['robot1_individual']}, 机器人2:{breakdown['robot2_individual']}")
        print(f"  总奖励   - 机器人1:{breakdown['robot1_total']}, 机器人2:{breakdown['robot2_total']}")
        
        # 显示状态向量信息
        if step % 40 == 0:  # 每40步显示一次状态向量
            robot1_state = state['robot1_state']
            robot2_state = state['robot2_state']
            print(f"  机器人1状态向量[10维]: [{', '.join([f'{x:.3f}' for x in robot1_state])}]")
            print(f"  机器人2状态向量[10维]: [{', '.join([f'{x:.3f}' for x in robot2_state])}]")
    
    if done:
        if info['persons_remaining'] == 0:
            if info['persons_dead'] == 0:
                print("所有人员成功疏散！")
            else:
                print(f"疏散结束 - {info['persons_dead']} 人死亡")
        else:
            print(f"达到最大步数，剩余 {info['persons_remaining']} 人存活")
        print(f"最终统计 - 疏散: {info['persons_escaped']}人, 死亡: {info['persons_dead']}人")
        print(f"最终疏散率: {info['evacuation_rate']:.2f}")
        print(f"总步数: {step + 1}")
        break

plt.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人环境演示文件
展示如何使用RobotEnvironment类进行火灾疏散仿真
"""

import random
from robot_env import RobotEnvironment

def demo_robot_environment():
    """演示机器人环境的基本使用"""
    
    # 初始化参数
    map_file = 'map.json'
    target_area = (3, 32, 2, 7)  # 人员初始区域
    num_persons = 150
    max_steps = 300
    
    # 创建机器人环境
    env = RobotEnvironment(map_file, target_area, num_persons, max_steps)
    
    # 设置机器人斥力因子（可调节的超参数）
    env.set_robot_repulsion_factor(0.4)  # 设置为火源斥力的40%
    
    print("=== 机器人环境演示 ===")
    print(f"初始人员数量: {env.initial_person_count}")
    print(f"机器人斥力因子: {env.robot_repulsion_factor}")
    print(f"机器人初始位置: {env.robot_positions}")
    print(f"火源位置: {list(env.map_loader.fires)}")
    print("机器人动作: 0=上, 1=下, 2=左, 3=右, 4=静止")
    print("-" * 50)
    
    # 仿真循环
    total_reward = 0
    step = 0
    
    while step < max_steps:
        # 随机生成机器人动作
        robot_actions = [random.randint(0, 4), random.randint(0, 4)]
        
        # 执行一步
        state, reward, done, info = env.step(robot_actions)
        
        total_reward += reward
        
        # 每10步打印一次状态
        if step % 10 == 0:
            print(f"步骤 {step:3d}: 机器人动作={robot_actions}, "
                  f"剩余{info['persons_remaining']:3d}人, "
                  f"疏散率={info['evacuation_rate']:.2f}, "
                  f"奖励={reward:.2f}")
        
        if done:
            break
        
        step += 1
    
    # 输出最终结果
    print("-" * 50)
    print("=== 仿真结束 ===")
    print(f"总步数: {step + 1}")
    print(f"最终疏散率: {info['evacuation_rate']:.2f}")
    print(f"剩余人数: {info['persons_remaining']}")
    print(f"总奖励: {total_reward:.2f}")
    
    if info['evacuation_rate'] == 1.0:
        print("✅ 所有人员成功疏散！")
    else:
        print("⚠️  仍有人员未能疏散")
    
    return env, total_reward

def test_different_repulsion_factors():
    """测试不同机器人斥力因子的效果"""
    
    print("\n=== 测试不同斥力因子 ===")
    factors = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    for factor in factors:
        env = RobotEnvironment('map.json', (3, 32, 2, 7), 150, 200)
        env.set_robot_repulsion_factor(factor)
        
        total_reward = 0
        step = 0
        
        while step < 200:
            robot_actions = [random.randint(0, 4), random.randint(0, 4)]
            state, reward, done, info = env.step(robot_actions)
            total_reward += reward
            
            if done:
                break
            step += 1
        
        print(f"斥力因子 {factor:.1f}: 疏散率={info['evacuation_rate']:.2f}, "
              f"步数={step+1:3d}, 总奖励={total_reward:.2f}")

if __name__ == "__main__":
    # 运行基本演示
    env, reward = demo_robot_environment()
    
    # 测试不同参数
    test_different_repulsion_factors()
    
    print("\n=== 环境接口说明 ===")
    print("RobotEnvironment主要方法:")
    print("- step(robot_actions): 执行一步仿真，返回(state, reward, done, info)")
    print("- reset(): 重置环境")
    print("- set_robot_repulsion_factor(factor): 设置机器人斥力因子") 
    print("- render(): 获取可视化地图")
    print("\nstate包含:")
    print("- person_positions: 人员位置列表")
    print("- fire_positions: 火源位置列表") 
    print("- robot_positions: 机器人位置列表")
    print("\ninfo包含:")
    print("- persons_remaining: 剩余人数")
    print("- persons_escaped: 已疏散人数")
    print("- evacuation_rate: 疏散成功率")
    print("- step: 当前步数")
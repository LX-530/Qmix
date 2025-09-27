# 多机器人火灾疏散环境 - QMIX强化学习

## 概要

`offpolicy\envs\CA\qmix_env.py`中可以修改环境超参数，最长时间步，最大人数，火灾区域，机器人斥力因子。如果想要打开仿真界面设置self.render = True
注意需要将此处的map_path替换为你的绝对地址！

`offpolicy/envs/CA/robot_env.py` 实现了一个火灾疏散环境，其中两个具有不同角色的机器人需要协作疏散被困人员：

主函数在`offpolicy\scripts\train\train.py`

所有超参数都在`offpolicy\config.py`文件中，调整训练episode数量在24行的num_env_steps。

模型会保存子啊`offpolicy\scripts\offpolicy`里面，按照训练时间保存，下面有如何加载模型的教程，
训练结果保存在`offpolicy\scripts\results`，如果测试模型记得修改offpolicy\envs\CA\qmix_env.py里面的超参数打开eval模式。


## 第一部分：环境介绍
- **机器人1（火源防护机器人）**：负责阻止人员靠近火源，减少伤亡
- **机器人2（出口疏散机器人）**：负责引导人员快速到达安全出口

### 状态空间

每个机器人都有一个**10维状态向量**，针对各自的角色任务进行了优化设计：

#### 机器人1状态向量（火源防护）
```python
state_vector = [
    rx_norm,           # [0] 归一化x坐标 
    ry_norm,           # [1] 归一化y坐标
    fire_distance,     # [2] 到最近火源的归一化距离
    fire_dir_x,        # [3] 火源方向x分量
    fire_dir_y,        # [4] 火源方向y分量
    nearby_ratio,      # [5] 周围3格内人员占比
    fire_danger_ratio, # [6] 火源3格内危险人员占比  
    avg_health_nearby, # [7] 周围人员平均健康值(0-1)
    min_health_nearby, # [8] 周围人员最低健康值(0-1)
    time_progress      # [9] 时间进度(0-1)
]
```

#### 机器人2状态向量（出口疏散）
```python
state_vector = [
    rx_norm,             # [0] 归一化x坐标
    ry_norm,             # [1] 归一化y坐标  
    exit_distance,       # [2] 到最近出口的归一化距离
    exit_dir_x,          # [3] 出口方向x分量
    exit_dir_y,          # [4] 出口方向y分量
    nearby_ratio,        # [5] 周围3格内人员占比
    exit_gathering_ratio,# [6] 出口5格内人员聚集占比
    avg_health_nearby,   # [7] 周围人员平均健康值(0-1)
    evacuation_rate,     # [8] 当前疏散率(0-1)
    time_progress        # [9] 时间进度(0-1)
]
```

### 动作空间

每个机器人都有**5个离散动作**：

```python
actions = {
    0: (-1, 0),  # 向上移动
    1: (1, 0),   # 向下移动  
    2: (0, -1),  # 向左移动
    3: (0, 1),   # 向右移动
    4: (0, 0)    # 停留不动
}
```

### 奖励机制 (Reward Function)

采用**混合奖励机制**，包含共享奖励和个体奖励：

#### 共享奖励（两个机器人都能获得）
1. **疏散奖励**：`evacuation_rate * 10 + time_efficiency * 2`
2. **健康保护奖励**：`(avg_health / 100) * 6`  
3. **协作效果奖励**：基于火源和出口附近的人员分布
4. **完成奖励**：
   - 完美疏散（全员逃脱）：20分
   - 任务完成但有伤亡：8分

#### 个体奖励

**机器人1（火源防护）**：
- 位置奖励：在火源3-5格范围内获得最高奖励
- 防护效果：成功驱赶火源附近人员的奖励
- 安全维护：火源周围危险人员越少奖励越高

**机器人2（出口疏散）**：
- 位置奖励：在出口2-4格范围内获得最高奖励  
- 疏散引导：成功引导人员到出口附近的奖励
- 疏散效率：基于疏散进度的动态奖励

### 环境动态特性

- **人员行为模型**：基于势场法的人员移动，考虑火源斥力和出口引力
- **火灾模型**：动态火源影响场，影响人员健康值  
- **健康系统**：人员健康值随火源接触时间递减
- **终止条件**：所有人员疏散完毕、死亡或达到最大步数(200步)

---

## 第二部分：QMIX算法集成


#### 支持的算法
- **QMIX**: RNN版本
- **mQMIX**: MLP版本

#### 网络架构

**策略网络**：
```python
# 每个agent的网络配置
- 输入维度: 10 (状态向量)  
- 隐藏层大小: 128 (可配置)
- 输出维度: 5 (动作空间)
- 网络类型: RNN + MLP (支持序列决策)
```

**QMIX混合网络**：
- 将各智能体Q值单调混合为全局Q值
- 保证个体最优策略与团队最优策略的一致性
- 支持中心化训练、分布式执行

#### 训练配置

**核心超参数**：
这些超参数都在`offpolicy\config.py`文件中可以找到。
```bash
--algorithm_name qmix          # 算法选择
--env_name two_robots         # 环境名称  
--num_env_steps 200000        # 训练步数
--episode_length 200          # 每回合最大步数
--buffer_size 5000           # 经验回放缓冲区大小
--hidden_size 128            # 网络隐层大小
--data_chunk_length 80       # RNN训练序列长度
```

**学习配置**：
```bash
--use_rnn_layer True         # 启用RNN层
--lr 0.0005                  # 学习率
--gamma 0.99                 # 折扣因子  
--tau 0.005                  # 软更新系数
--use_double_q True          # 双Q网络
```

### 分布式训练支持

- **环境并行**：支持多环境并行数据收集
- **经验重放**：优先经验重放(PER)可选
- **模型保存**：定期保存训练好的模型参数

---

## 第三部分：使用方法和配置


### 高级配置

#### 环境自定义
```python
# 在offpolicy/envs/CA/qmix_env.py中调整
target_area = (3, 32, 2, 7)   # 人员初始区域 
num_persons = 150             # 人员数量
max_steps = 200              # 最大步数
map_path = "path/to/map.json" # 地图文件路径
```

#### 网络架构调整
```bash
--hidden_size 256            # 增大网络容量
--use_conv1d True           # 启用卷积层  
--attn True                 # 启用注意力机制
--use_orthogonal_init True  # 正交初始化
```

#### 训练优化
```bash
--n_rollout_threads 8       # 并行环境数
--use_value_active_masks True  # 使用价值掩码
--use_policy_active_masks True # 使用策略掩码  
--use_per True              # 优先经验重放
--per_alpha 0.6             # PER alpha参数
```
#### 评估模型

在训练好模型后，可以加载模型参数进行训练。需要在`offpolicy\config.py`中193行加入模型地址，输入绝对路径如`F:\1Business_code\20250910-qmix_fire\qmix\offpolicy\scripts\offpolicy\models2025-09-13-19-08-25\qmix_models`

还需要将`offpolicy\envs\CA\qmix_env.py`中设置self.eval = True
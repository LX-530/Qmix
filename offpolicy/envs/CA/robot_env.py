import numpy as np
import random
import matplotlib.pyplot as plt
from offpolicy.envs.CA.map_loader import MapLoader
from offpolicy.envs.CA.person_behavior import Person
from offpolicy.envs.CA.fire_model import FireModel

class RobotEnvironment:
    def __init__(self, map_file, target_area, num_persons, max_steps=300):
        self.map_loader = MapLoader(map_file)
        self.target_area = target_area
        self.num_persons = num_persons
        self.max_steps = max_steps
        self.current_step = 0
        
        self.robot_positions = []
        self.robot_repulsion_factor = 0.5
        
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay
        }
        
        self._reset_environment()
    
    def _reset_environment(self):
        self.current_step = 0
        self._initialize_persons()
        self._initialize_robots()
        self.fire_model = FireModel(self.map_loader.fires, self.map_loader.rows, self.map_loader.cols)
        self.initial_person_count = len(self.persons)
        self.escaped_persons = 0
        
        return self._get_state()
    
    
    def _initialize_persons(self):
        target_empty_positions = []
        for r in range(self.target_area[0], self.target_area[1]):
            for c in range(self.target_area[2], self.target_area[3]):
                if self.map_loader.map_data[r, c] == 0:
                    target_empty_positions.append((r, c))
        
        self.persons = []
        for _ in range(min(self.num_persons, len(target_empty_positions))):
            pos = random.choice(target_empty_positions)
            target_empty_positions.remove(pos)
            self.persons.append(Person(pos, self.map_loader))
    
    def _initialize_robots(self):
        """根据角色初始化机器人位置"""
        empty_positions = []
        for r in range(self.map_loader.rows):
            for c in range(self.map_loader.cols):
                if self.map_loader.map_data[r, c] == 0:
                    occupied = any(person.position == (r, c) for person in self.persons)
                    if not occupied:
                        empty_positions.append((r, c))
        
        self.robot_positions = []
        
        # 机器人1：火源防护机器人 - 初始化在火源附近
        if self.map_loader.fires and len(empty_positions) > 0:
            fire_pos = list(self.map_loader.fires)[0]  # 选择第一个火源
            robot1_pos = self._find_best_position_near_target(fire_pos, empty_positions, 3, 6)
            if robot1_pos:
                self.robot_positions.append(robot1_pos)
                empty_positions.remove(robot1_pos)
        
        # 机器人2：出口疏散机器人 - 初始化在出口附近
        if self.map_loader.exits and len(empty_positions) > 0:
            exit_pos = list(self.map_loader.exits)[0]  # 选择第一个出口
            robot2_pos = self._find_best_position_near_target(exit_pos, empty_positions, 2, 4)
            if robot2_pos:
                self.robot_positions.append(robot2_pos)
                empty_positions.remove(robot2_pos)
        
        # 如果没有找到合适位置，随机选择
        while len(self.robot_positions) < 2 and empty_positions:
            pos = random.choice(empty_positions)
            self.robot_positions.append(pos)
            empty_positions.remove(pos)
    
    def _find_best_position_near_target(self, target, available_positions, min_dist, max_dist):
        """在目标附近找到最佳位置"""
        tx, ty = target
        candidates = []
        
        # 寻找在理想距离范围内的位置
        for pos in available_positions:
            px, py = pos
            dist = max(abs(px - tx), abs(py - ty))
            if min_dist <= dist <= max_dist:
                candidates.append((pos, dist))
        
        if candidates:
            # 选择距离最接近理想值的位置（理想值为min_dist + 1）
            ideal_dist = min_dist + 1
            best_pos = min(candidates, key=lambda x: abs(x[1] - ideal_dist))[0]
            return best_pos
        
        # 如果没有理想位置，选择最近的可用位置
        if available_positions:
            nearest_pos = min(available_positions, 
                            key=lambda pos: max(abs(pos[0] - tx), abs(pos[1] - ty)))
            return nearest_pos
            
        return None
    
    def _compute_robot_repulsion_field(self):
        robot_field = np.zeros((self.map_loader.rows, self.map_loader.cols))
        
        A = self.fire_model.A * self.robot_repulsion_factor
        B = self.fire_model.B
        cutoff_radius = self.fire_model.cutoff_radius
        
        for i in range(self.map_loader.rows):
            for j in range(self.map_loader.cols):
                for rx, ry in self.robot_positions:
                    dist = np.sqrt((i - rx)**2 + (j - ry)**2)
                    if 0 < dist <= cutoff_radius:
                        robot_field[i, j] += A * np.exp(-B * dist)
                    elif dist == 0:
                        robot_field[i, j] = np.inf
        
        return robot_field
    
    def _is_valid_position(self, pos):
        r, c = pos
        if r < 0 or r >= self.map_loader.rows or c < 0 or c >= self.map_loader.cols:
            return False
        if self.map_loader.map_data[r, c] == 1:  # 墙壁
            return False
        if (r, c) in self.map_loader.fires:  # 火源位置
            return False
        return True
    
    def step(self, robot_actions):
        if self.current_step >= self.max_steps:
            # 当达到最大步数时，也要提供完整的info结构
            alive_persons_list = [p for p in self.persons if not p.is_dead]
            dead_persons = len([p for p in self.persons if p.is_dead])
            avg_health = sum(p.health for p in alive_persons_list) / len(alive_persons_list) if alive_persons_list else 0.0
            
            info = {
                "step": self.current_step,
                "persons_remaining": len(alive_persons_list),
                "persons_escaped": self.escaped_persons,
                "persons_dead": dead_persons,
                "evacuation_rate": self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0,
                "avg_health": avg_health,
                "reason": "max_steps_reached"
            }
            return self._get_state(), [0, 0], True, info
        
        self._update_robot_positions(robot_actions)
        
        fire_field = self.fire_model.compute_dynamic_field()
        robot_field = self._compute_robot_repulsion_field()
        combined_dynamic_field = fire_field + robot_field
        
        self._update_person_positions(combined_dynamic_field)
        
        # 更新所有人员的健康值
        self._update_persons_health()
        
        reward = self._calculate_reward()
        
        # 修改结束条件：所有人员死亡或撤离，或达到最大步数
        alive_persons = [p for p in self.persons if not p.is_dead]
        done = len(alive_persons) == 0 or self.current_step >= self.max_steps
        
        self.current_step += 1
        
        dead_persons = len([p for p in self.persons if p.is_dead])
        alive_persons_list = [p for p in self.persons if not p.is_dead]
        alive_persons_count = len(alive_persons_list)
        
        # 计算当前时间步所有人员的平均健康值
        avg_health = sum(p.health for p in alive_persons_list) / len(alive_persons_list) if alive_persons_list else 0.0
        
        info = {
            "step": self.current_step,
            "persons_remaining": alive_persons_count,
            "persons_escaped": self.escaped_persons,
            "persons_dead": dead_persons,
            "evacuation_rate": self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0,
            "avg_health": avg_health
        }
        
        return self._get_state(), reward, done, info
    
    def _update_robot_positions(self, robot_actions):
        new_positions = []
        for i, action in enumerate(robot_actions):
            if i >= len(self.robot_positions):
                break
                
            current_pos = self.robot_positions[i]
            dr, dc = self.actions[action]
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)
            
            if self._is_valid_position(new_pos) and new_pos not in new_positions:
                new_positions.append(new_pos)
            else:
                new_positions.append(current_pos)
        
        self.robot_positions = new_positions
    
    def _update_person_positions(self, dynamic_field):
        next_occupancy = np.zeros((self.map_loader.rows, self.map_loader.cols), dtype=int)
        
        for rx, ry in self.robot_positions:
            next_occupancy[rx, ry] = 1
        
        next_positions = {}
        
        for person in self.persons:
            if not person.escaped and not person.is_dead:
                next_pos = person.get_possible_moves(next_occupancy, dynamic_field)
                if next_pos in next_positions:
                    if random.random() > 0.5:
                        next_positions[next_pos] = person
                else:
                    next_positions[next_pos] = person
                    next_occupancy[next_pos[0], next_pos[1]] = 1
        
        escaped = []
        for person in self.persons:
            if person in next_positions.values():
                person.position = [pos for pos, p in next_positions.items() if p == person][0]
                if person.position in self.map_loader.exits:
                    person.escaped = True
                    escaped.append(person)
                    self.escaped_persons += 1
        
        for p in escaped:
            self.persons.remove(p)
    
    def _update_persons_health(self):
        """更新所有人员的健康值"""
        for person in self.persons:
            if not person.escaped and not person.is_dead:
                person.update_health(self.map_loader.fires)
    
    def _calculate_reward(self):
        """计算双机器人个体奖励"""
        alive_persons = [p for p in self.persons if not p.is_dead]
        
        # 共享奖励部分
        evacuation_rate = self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0
        time_efficiency = max(0, (self.max_steps - self.current_step) / self.max_steps)
        shared_evacuation_reward = evacuation_rate * 10 + time_efficiency * 2
        
        # 健康保护奖励（共享）
        if alive_persons:
            avg_health = sum(p.health for p in alive_persons) / len(alive_persons)
            shared_health_reward = (avg_health / 100) * 6
        else:
            shared_health_reward = 6
        
        # 协作效果奖励（共享）
        shared_cooperation_reward = self._calculate_cooperation_reward(alive_persons)
        
        # 完成奖励（共享）
        shared_completion_bonus = 0
        if len(alive_persons) == 0:
            if self.escaped_persons == self.initial_person_count:
                shared_completion_bonus = 20  # 完美撤离
            else:
                shared_completion_bonus = 8   # 至少结束了
        
        # 计算每个机器人的个体奖励
        robot1_reward, robot2_reward = self._calculate_individual_rewards(alive_persons)
        
        # 总奖励 = 共享奖励 + 个体奖励
        total_robot1_reward = (shared_evacuation_reward + shared_health_reward + 
                              shared_cooperation_reward + shared_completion_bonus + robot1_reward)
        total_robot2_reward = (shared_evacuation_reward + shared_health_reward + 
                              shared_cooperation_reward + shared_completion_bonus + robot2_reward)
        
        # 存储奖励分解供调试使用
        self.reward_breakdown = {
            "shared_evacuation": round(shared_evacuation_reward, 3),
            "shared_health": round(shared_health_reward, 3),
            "shared_cooperation": round(shared_cooperation_reward, 3),
            "shared_completion": round(shared_completion_bonus, 3),
            "robot1_individual": round(robot1_reward, 3),
            "robot2_individual": round(robot2_reward, 3),
            "robot1_total": round(total_robot1_reward, 3),
            "robot2_total": round(total_robot2_reward, 3)
        }
        
        return [total_robot1_reward, total_robot2_reward]
    
    def _calculate_individual_rewards(self, alive_persons):
        """计算每个机器人的个体奖励"""
        robot1_reward = 0
        robot2_reward = 0
        
        if len(self.robot_positions) >= 2:
            # 机器人1（火源防护）个体奖励
            robot1_reward = self._calculate_robot1_individual_reward(alive_persons)
            
            # 机器人2（出口疏散）个体奖励  
            robot2_reward = self._calculate_robot2_individual_reward(alive_persons)
        
        return robot1_reward, robot2_reward
    
    def _calculate_robot1_individual_reward(self, alive_persons):
        """计算机器人1的个体奖励"""
        reward = 0
        robot1_pos = self.robot_positions[0]
        
        # 1. 位置奖励 - 是否在火源附近的合适位置
        if self.map_loader.fires:
            min_fire_dist = min([max(abs(robot1_pos[0] - fx), abs(robot1_pos[1] - fy)) 
                               for fx, fy in self.map_loader.fires])
            # 距离火源3-5格最佳，给予位置奖励
            if 3 <= min_fire_dist <= 5:
                reward += 3
            elif 2 <= min_fire_dist <= 6:
                reward += 1
        
        # 2. 火源防护效果奖励
        persons_near_fire = 0
        persons_driven_away = 0  # 被驱赶远离火源的人员
        
        for person in alive_persons:
            px, py = person.position
            # 统计火源3格内的危险人员
            for fx, fy in self.map_loader.fires:
                if max(abs(px - fx), abs(py - fy)) <= 3:
                    persons_near_fire += 1
                    break
            
            # 统计机器人3格内但远离火源的人员（防护成功）
            robot_dist = max(abs(px - robot1_pos[0]), abs(py - robot1_pos[1]))
            if robot_dist <= 3:
                fire_dist = min([max(abs(px - fx), abs(py - fy)) 
                               for fx, fy in self.map_loader.fires]) if self.map_loader.fires else float('inf')
                if fire_dist > 4:  # 人员在机器人附近但远离火源
                    persons_driven_away += 1
        
        # 火源附近人员越少越好
        total_persons = max(len(alive_persons), 1)
        fire_safety_ratio = 1 - (persons_near_fire / total_persons)
        reward += fire_safety_ratio * 4
        
        # 成功驱赶人员的奖励
        drive_away_ratio = persons_driven_away / total_persons
        reward += drive_away_ratio * 2
        
        return reward
    
    def _calculate_robot2_individual_reward(self, alive_persons):
        """计算机器人2的个体奖励"""
        reward = 0
        robot2_pos = self.robot_positions[1]
        
        # 1. 位置奖励 - 是否在出口附近的合适位置
        if self.map_loader.exits:
            min_exit_dist = min([max(abs(robot2_pos[0] - ex), abs(robot2_pos[1] - ey)) 
                               for ex, ey in self.map_loader.exits])
            # 距离出口2-4格最佳，给予位置奖励
            if 2 <= min_exit_dist <= 4:
                reward += 3
            elif 1 <= min_exit_dist <= 5:
                reward += 1
        
        # 2. 疏散引导效果奖励
        persons_near_exit = 0
        persons_guided = 0  # 被引导到出口的人员
        
        for person in alive_persons:
            px, py = person.position
            # 统计出口5格内的人员
            for ex, ey in self.map_loader.exits:
                if max(abs(px - ex), abs(py - ey)) <= 5:
                    persons_near_exit += 1
                    break
            
            # 统计机器人3格内且接近出口的人员（引导成功）
            robot_dist = max(abs(px - robot2_pos[0]), abs(py - robot2_pos[1]))
            if robot_dist <= 3:
                exit_dist = min([max(abs(px - ex), abs(py - ey)) 
                               for ex, ey in self.map_loader.exits]) if self.map_loader.exits else float('inf')
                if exit_dist <= 6:  # 人员在机器人附近且接近出口
                    persons_guided += 1
        
        # 出口附近人员越多越好（说明疏散进行中）
        total_persons = max(len(alive_persons), 1)
        exit_gathering_ratio = persons_near_exit / total_persons
        reward += exit_gathering_ratio * 4
        
        # 成功引导人员的奖励
        guide_ratio = persons_guided / total_persons
        reward += guide_ratio * 2
        
        # 疏散进度奖励
        evacuation_progress = self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0
        reward += evacuation_progress * 2
        
        return reward
    
    def _calculate_position_reward(self):
        """计算机器人位置奖励"""
        if len(self.robot_positions) < 2:
            return 0
            
        reward = 0
        
        # 机器人1（火源防护）应该靠近火源
        robot1_pos = self.robot_positions[0]
        if self.map_loader.fires:
            min_fire_dist = min([max(abs(robot1_pos[0] - fx), abs(robot1_pos[1] - fy)) 
                               for fx, fy in self.map_loader.fires])
            # 距离火源2-4格最佳
            if 2 <= min_fire_dist <= 4:
                reward += 3
            elif min_fire_dist <= 6:
                reward += 1
        
        # 机器人2（出口疏散）应该靠近出口
        robot2_pos = self.robot_positions[1]  
        if self.map_loader.exits:
            min_exit_dist = min([max(abs(robot2_pos[0] - ex), abs(robot2_pos[1] - ey)) 
                               for ex, ey in self.map_loader.exits])
            # 距离出口1-3格最佳
            if 1 <= min_exit_dist <= 3:
                reward += 3
            elif min_exit_dist <= 5:
                reward += 1
                
        return reward
    
    def _calculate_cooperation_reward(self, alive_persons):
        """计算协作效果奖励"""
        if not alive_persons or len(self.robot_positions) < 2:
            return 0
            
        reward = 0
        
        # 统计火源附近和出口附近的人员
        persons_near_fire = 0
        persons_near_exit = 0
        
        for person in alive_persons:
            px, py = person.position
            
            # 检查是否靠近火源
            for fx, fy in self.map_loader.fires:
                if max(abs(px - fx), abs(py - fy)) <= 3:
                    persons_near_fire += 1
                    break
            
            # 检查是否靠近出口
            for ex, ey in self.map_loader.exits:
                if max(abs(px - ex), abs(py - ey)) <= 5:
                    persons_near_exit += 1
                    break
        
        # 火源附近人员越少越好
        if persons_near_fire == 0:
            reward += 4
        elif persons_near_fire <= 2:
            reward += 2
        
        # 出口附近人员较多说明疏散进行中
        if persons_near_exit >= len(alive_persons) * 0.3:  # 30%以上的人在出口附近
            reward += 3
        elif persons_near_exit >= len(alive_persons) * 0.1:  # 10%以上
            reward += 1
            
        return reward
    
    def _get_state(self):
        """为两个机器人提供10维一维向量状态"""
        alive_persons = [p for p in self.persons if not p.escaped and not p.is_dead]
        
        # 机器人1状态：火源防护机器人 [10维]
        robot1_state = self._get_robot1_state_vector(alive_persons)
        
        # 机器人2状态：出口疏散机器人 [10维]
        robot2_state = self._get_robot2_state_vector(alive_persons)
        global_info = {
            "step": self.current_step,
            "total_persons": len(alive_persons),
            "avg_health": sum(p.health for p in alive_persons) / len(alive_persons) if alive_persons else 100
        }
        
        return {
            "robot1_state": robot1_state,
            "robot2_state": robot2_state,

        }
    
    def _get_robot_state(self, robot_id, alive_persons, fire_positions, exit_positions, role):
        """获取特定机器人的状态"""
        if robot_id >= len(self.robot_positions):
            return {}
            
        robot_pos = self.robot_positions[robot_id]
        
        # 基础状态
        state = {
            "position": robot_pos,
            "role": role
        }
        
        # 周围人员信息（5x5范围内）
        nearby_persons = []
        for person in alive_persons:
            px, py = person.position
            rx, ry = robot_pos
            if abs(px - rx) <= 2 and abs(py - ry) <= 2:
                nearby_persons.append({
                    "position": person.position,
                    "health": person.health,
                    "distance_to_robot": max(abs(px - rx), abs(py - ry))
                })
        
        state["nearby_persons"] = nearby_persons
        
        if role == "fire_guard":
            # 火源防护机器人关注：
            # 1. 到最近火源的距离
            # 2. 火源周围的人员数量
            # 3. 危险区域的人员健康状况
            min_fire_dist = min([max(abs(robot_pos[0] - fx), abs(robot_pos[1] - fy)) 
                               for fx, fy in fire_positions]) if fire_positions else float('inf')
            
            # 统计火源3格内的人员
            persons_near_fire = 0
            for person in alive_persons:
                px, py = person.position
                for fx, fy in fire_positions:
                    if max(abs(px - fx), abs(py - fy)) <= 3:
                        persons_near_fire += 1
                        break
            
            state.update({
                "distance_to_nearest_fire": min_fire_dist,
                "persons_near_fire": persons_near_fire,
                "fire_positions": fire_positions
            })
            
        elif role == "exit_guide":
            # 出口疏散机器人关注：
            # 1. 到最近出口的距离
            # 2. 出口周围的人员数量
            # 3. 疏散效率
            min_exit_dist = min([max(abs(robot_pos[0] - ex), abs(robot_pos[1] - ey)) 
                               for ex, ey in exit_positions]) if exit_positions else float('inf')
            
            # 统计出口5格内的人员
            persons_near_exit = 0
            for person in alive_persons:
                px, py = person.position
                for ex, ey in exit_positions:
                    if max(abs(px - ex), abs(py - ey)) <= 5:
                        persons_near_exit += 1
                        break
                        
            state.update({
                "distance_to_nearest_exit": min_exit_dist,
                "persons_near_exit": persons_near_exit,
                "exit_positions": exit_positions
            })
        
        return state
    
    def _get_robot1_state_vector(self, alive_persons):
        """获取机器人1（火源防护）的10维状态向量"""
        if len(self.robot_positions) < 1:
            return [0.0] * 10
            
        robot_pos = self.robot_positions[0]
        rx, ry = robot_pos
        
        # 归一化参数
        max_distance = max(self.map_loader.rows, self.map_loader.cols)
        total_persons = max(len(alive_persons), 1)  # 避免除零
        
        state_vector = []
        
        # 1-2. 基础位置信息 (归一化坐标)
        state_vector.extend([
            rx / self.map_loader.rows,      # 归一化x坐标
            ry / self.map_loader.cols       # 归一化y坐标
        ])
        
        # 3-5. 火源相关信息
        if self.map_loader.fires:
            # 找到最近火源
            min_fire_dist = float('inf')
            nearest_fire = None
            for fx, fy in self.map_loader.fires:
                dist = max(abs(rx - fx), abs(ry - fy))
                if dist < min_fire_dist:
                    min_fire_dist = dist
                    nearest_fire = (fx, fy)
            
            # 到最近火源的距离和方向
            fire_distance = min_fire_dist / max_distance
            if nearest_fire and min_fire_dist > 0:
                fx, fy = nearest_fire
                fire_dir_x = (fx - rx) / min_fire_dist
                fire_dir_y = (fy - ry) / min_fire_dist
            else:
                fire_dir_x, fire_dir_y = 0.0, 0.0
        else:
            fire_distance, fire_dir_x, fire_dir_y = 1.0, 0.0, 0.0
            
        state_vector.extend([fire_distance, fire_dir_x, fire_dir_y])
        
        # 6-9. 人员相关信息
        nearby_persons_count = 0
        nearby_healths = []
        persons_near_fire = 0
        
        for person in alive_persons:
            px, py = person.position
            
            # 统计周围3格内人员
            if max(abs(px - rx), abs(py - ry)) <= 3:
                nearby_persons_count += 1
                nearby_healths.append(person.health)
            
            # 统计火源3格内人员
            for fx, fy in self.map_loader.fires:
                if max(abs(px - fx), abs(py - fy)) <= 3:
                    persons_near_fire += 1
                    break
        
        # 人员统计信息
        nearby_ratio = nearby_persons_count / total_persons
        fire_danger_ratio = persons_near_fire / total_persons
        avg_health_nearby = (sum(nearby_healths) / len(nearby_healths) / 100) if nearby_healths else 1.0
        min_health_nearby = (min(nearby_healths) / 100) if nearby_healths else 1.0
        
        state_vector.extend([nearby_ratio, fire_danger_ratio, avg_health_nearby, min_health_nearby])
        
        # 10. 时间进度
        time_progress = self.current_step / self.max_steps
        state_vector.append(time_progress)
        
        # 格式化到小数点后三位
        state_vector = [round(float(x), 3) for x in state_vector]
        
        return state_vector
    
    def _get_robot2_state_vector(self, alive_persons):
        """获取机器人2（出口疏散）的10维状态向量"""
        if len(self.robot_positions) < 2:
            return [0.0] * 10
            
        robot_pos = self.robot_positions[1]
        rx, ry = robot_pos
        
        # 归一化参数
        max_distance = max(self.map_loader.rows, self.map_loader.cols)
        total_persons = max(len(alive_persons), 1)  # 避免除零
        
        state_vector = []
        
        # 1-2. 基础位置信息 (归一化坐标)
        state_vector.extend([
            rx / self.map_loader.rows,      # 归一化x坐标
            ry / self.map_loader.cols       # 归一化y坐标
        ])
        
        # 3-5. 出口相关信息
        if self.map_loader.exits:
            # 找到最近出口
            min_exit_dist = float('inf')
            nearest_exit = None
            for ex, ey in self.map_loader.exits:
                dist = max(abs(rx - ex), abs(ry - ey))
                if dist < min_exit_dist:
                    min_exit_dist = dist
                    nearest_exit = (ex, ey)
            
            # 到最近出口的距离和方向
            exit_distance = min_exit_dist / max_distance
            if nearest_exit and min_exit_dist > 0:
                ex, ey = nearest_exit
                exit_dir_x = (ex - rx) / min_exit_dist
                exit_dir_y = (ey - ry) / min_exit_dist
            else:
                exit_dir_x, exit_dir_y = 0.0, 0.0
        else:
            exit_distance, exit_dir_x, exit_dir_y = 1.0, 0.0, 0.0
            
        state_vector.extend([exit_distance, exit_dir_x, exit_dir_y])
        
        # 6-9. 人员相关信息
        nearby_persons_count = 0
        nearby_healths = []
        persons_near_exit = 0
        
        for person in alive_persons:
            px, py = person.position
            
            # 统计周围3格内人员
            if max(abs(px - rx), abs(py - ry)) <= 3:
                nearby_persons_count += 1
                nearby_healths.append(person.health)
            
            # 统计出口5格内人员
            for ex, ey in self.map_loader.exits:
                if max(abs(px - ex), abs(py - ey)) <= 5:
                    persons_near_exit += 1
                    break
        
        # 人员统计信息
        nearby_ratio = nearby_persons_count / total_persons
        exit_gathering_ratio = persons_near_exit / total_persons
        avg_health_nearby = (sum(nearby_healths) / len(nearby_healths) / 100) if nearby_healths else 1.0
        evacuation_rate = self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0
        
        state_vector.extend([nearby_ratio, exit_gathering_ratio, avg_health_nearby, evacuation_rate])
        
        # 10. 时间进度
        time_progress = self.current_step / self.max_steps
        state_vector.append(time_progress)
        
        # 格式化到小数点后三位
        state_vector = [round(float(x), 3) for x in state_vector]
        
        return state_vector
    
    def reset(self):
        self._reset_environment()
        return self._get_state()
    
    def set_robot_repulsion_factor(self, factor):
        self.robot_repulsion_factor = factor
    
    def render(self, mode='human'):
        """渲染环境，进行可视化显示"""
        vis_map = np.copy(self.map_loader.map_data)
        
        # 添加人员标记（区分活着和死亡）
        for person in self.persons:
            x, y = person.position
            if person.is_dead:
                vis_map[x, y] = 6  # 死亡人员标记
            else:
                vis_map[x, y] = 4  # 活着的人员标记
        
        # 添加火源标记
        for fx, fy in self.map_loader.fires:
            vis_map[fx, fy] = 3  # Fire
        
        # 添加机器人标记
        for rx, ry in self.robot_positions:
            vis_map[rx, ry] = 5  # Robot marker
        
        if mode == 'human':
            # 执行可视化显示
            plt.imshow(vis_map, cmap='hot', interpolation='nearest')
            plt.title(f'Step {self.current_step}')
            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()
        elif mode == 'rgb_array':
            # 返回vis_map数组（用于其他用途）
            return vis_map
        
        return vis_map
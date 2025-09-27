# person_behavior.py
import random
import numpy as np

class Person:
    
    def __init__(self, position, map_loader):
        self.position = position
        self.escaped = False
        self.map_loader = map_loader
        self.health = 100  # 初始健康值
        self.is_dead = False  # 死亡状态

    def get_possible_moves(self, occupancy, dynamic_field):
        x, y = self.position
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        possible_moves = []
        min_field = np.inf

        total_field = self.map_loader.static_field + dynamic_field

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.map_loader.rows and 0 <= ny < self.map_loader.cols and
                self.map_loader.map_data[nx, ny] != 1 and occupancy[nx, ny] == 0):  # Not obstacle, not occupied
                field_value = total_field[nx, ny]
                if field_value < min_field:
                    min_field = field_value
                    possible_moves = [(nx, ny)]
                elif field_value == min_field:
                    possible_moves.append((nx, ny))

        # Include staying put if no better move, but prefer moving
        stay_field = total_field[x, y]
        if possible_moves:
            return random.choice(possible_moves)
        else:
            return self.position  # Stay if no move
    
    def update_health(self, fire_positions):
        """根据与火源的距离更新健康值"""
        if self.is_dead or self.escaped:
            return
        
        x, y = self.position
        min_fire_distance = float('inf')
        
        # 计算到最近火源的距离
        for fx, fy in fire_positions:
            distance = max(abs(x - fx), abs(y - fy))  # 使用切比雪夫距离（棋盘距离）
            min_fire_distance = min(min_fire_distance, distance)
        
        # 根据距离扣除健康值
        if min_fire_distance == 2:
            self.health -= 10
        elif min_fire_distance == 3:
            self.health -= 5
        
        # 检查是否死亡
        if self.health <= 0:
            self.health = 0
            self.is_dead = True
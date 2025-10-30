# Boid Model

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import copy

MAX_X = 400.0
MAX_Y = 400.0
R = 10.0
DEL_T = 0.01
TIME_STEP = 1000
AGENT_NUMBER = 100

class Boid:
    def __init__(self, id):
        self.id = id
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def initialize(self, max_x=MAX_X, max_y=MAX_Y):
        self.x = random.random() * max_x
        self.y = random.random() * max_y
        self.vx = random.random() * max_x / 100.0
        self.vy = random.random() * max_y / 100.0

    def isNearByBoid(self, boid):
        distance = math.sqrt((self.x - boid.x)**2 + (self.y - boid.y)**2)
        return True if distance <= R else False
    
    # 相互作用を変更する場合はこの関数を変える
    def changeVelocity(self, boids):
        vectorized_func = np.vectorize(self.isNearByBoid, otypes=[bool])
        near_by_mask = vectorized_func(boids)
        near_by_boids = boids[near_by_mask]
        near_by_vxs = np.array([boid.vx for boid in near_by_boids])
        near_by_vys = np.array([boid.vy for boid in near_by_boids])
        new_vx = (self.vx + near_by_vxs.sum()) / (1 + len(near_by_boids))
        new_vy = (self.vy + near_by_vys.sum()) / (1 + len(near_by_boids))
        self.vx = new_vx
        self.vy = new_vy

    def updatePosition(self, del_t=DEL_T, max_x=MAX_X, max_y=MAX_Y):
        # 周期境界条件
        self.x += (self.vx * del_t) % max_x
        self.y += (self.vy * del_t) % max_y

def main():
    boids = np.array([Boid(idx) for idx in range(AGENT_NUMBER)])
    for i in range(AGENT_NUMBER):
        boids[i].initialize()
    
    for t in range(TIME_STEP):
        current_boids = copy.deepcopy(boids)
        for i in range(AGENT_NUMBER):
            boids[i].changeVelocity(current_boids)
            boids[i].updatePosition()

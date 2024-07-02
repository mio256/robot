import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定数の定義
SEPARATION_DISTANCE = 3.5
ALIGNMENT_DISTANCE = 7.0
COHESION_DISTANCE = 5.0

SEPARATION_WEIGHT = 1.75
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0

MAX_SPEED = 1.0
SPACE_SIZE = 50

# 初期座標と初期速度を指定
initial_positions = [
    [-2, 6],
    [4, 4],
    [0, 2],
    [-6, 0],
    [6, 0],
    [0, -2],
    [-8, -4],
    [-4, -6],
    [2, -6],
    [8, -6]
]

# 初期座標の中心が(0, 0)になるようにSPACE_SIZE/2を足す
initial_positions = [[pos[0] + SPACE_SIZE / 2, pos[1] + SPACE_SIZE / 2] for pos in initial_positions]

initial_velocities = [
    [-np.sqrt(2), np.sqrt(2)],
    [np.sqrt(2), np.sqrt(2)],
    [0, 1],
    [np.sqrt(2), np.sqrt(2)],
    [-np.sqrt(2), np.sqrt(2)],
    [np.sqrt(2), np.sqrt(2)],
    [0, 1],
    [np.sqrt(2), np.sqrt(2)],
    [-np.sqrt(2), np.sqrt(2)]
]

# Boidクラスの定義


class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64) * MAX_SPEED

    def update(self, boids):
        separation = np.zeros(2, dtype=np.float64)
        alignment = np.zeros(2, dtype=np.float64)
        cohesion = np.zeros(2, dtype=np.float64)
        count = 0

        for boid in boids:
            if boid is not self:
                distance = np.linalg.norm(boid.position - self.position)
                if distance < SEPARATION_DISTANCE:
                    separation -= (boid.position - self.position)
                if distance < ALIGNMENT_DISTANCE:
                    alignment += boid.velocity
                if distance < COHESION_DISTANCE:
                    cohesion += boid.position
                    count += 1

        if count > 0:
            alignment /= count
            alignment = (alignment / np.linalg.norm(alignment)) * MAX_SPEED
            cohesion /= count
            cohesion = cohesion - self.position

        self.velocity += (separation * SEPARATION_WEIGHT +
                          alignment * ALIGNMENT_WEIGHT +
                          cohesion * COHESION_WEIGHT)

        if np.linalg.norm(self.velocity) > MAX_SPEED:
            self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * MAX_SPEED

        self.position += self.velocity
        self.position %= SPACE_SIZE


# 初期座標と速度を持つボイドを作成
boids = [Boid(position=pos, velocity=vel) for pos, vel in zip(initial_positions, initial_velocities)]

fig, ax = plt.subplots()
scat = ax.scatter([boid.position[0] for boid in boids],
                  [boid.position[1] for boid in boids])


def init():
    scat.set_offsets([boid.position for boid in boids])
    return scat,


def update(frame):
    for boid in boids:
        boid.update(boids)
    scat.set_offsets([boid.position for boid in boids])
    return scat,


ani = FuncAnimation(fig, update, frames=200, init_func=init, interval=500, blit=False)
plt.xlim(0, SPACE_SIZE)
plt.ylim(0, SPACE_SIZE)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

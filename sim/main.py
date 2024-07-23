from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time

# Define constants for Boid behavior
SEPARATION_DISTANCE = 0.35
ALIGNMENT_DISTANCE = 0.7
COHESION_DISTANCE = 0.5

SEPARATION_WEIGHT = 0.175
ALIGNMENT_WEIGHT = 0.15 
COHESION_WEIGHT = 0.1

MAX_SPEED = 0.01
SPACE_SIZE = np.array([2, 2])

# Initial positions and velocities
initial_positions = [
    [-2, 6, 0],
    [4, 4, 0],
    [0, 2, 0],
    [-6, 0, 0],
    [6, 0, 0],
    [0, -2, 0],
    [-8, -4, 0],
    [-4, -6, 0],
    [2, -6, 0],
    [8, -6, 0]
]

for i in range(len(initial_positions)):
    initial_positions[i] = [j/10 for j in initial_positions[i]]

initial_velocities = [
    [-1/np.sqrt(2), 1/np.sqrt(2)],
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [0, 1],
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [-1/np.sqrt(2), 1/np.sqrt(2)],
    [-1/np.sqrt(2), 1/np.sqrt(2)],
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [0, 1],
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [-1/np.sqrt(2), 1/np.sqrt(2)]
]

initial_velocities = [np.array(vel) * 0.01 for vel in initial_velocities]

# Boid class definition


class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position[:2], dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)

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


# Create Boids with initial positions and velocities
boids = [Boid(position=pos, velocity=vel) for pos, vel in zip(initial_positions, initial_velocities)]

# クライアントの作成とシミュレーションの開始
client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)

# Kilobotのハンドルを取得
kilobots = []
all_objects = sim.getObjectsInTree(sim.handle_scene)
for obj in all_objects:
    name = sim.getObjectAlias(obj)
    if name == 'KilobotD':
        kilobots.append(obj)

if len(kilobots) != 10:
    print(f"Expected 10 KilobotD objects, but found {len(kilobots)}")
else:
    print("Successfully found 10 KilobotD objects")

# シミュレーションの開始
sim.startSimulation()

# 各Kilobotの座標を設定
for i, kilobot in enumerate(kilobots):
    sim.setObjectPosition(kilobot, -1, initial_positions[i])

# シミュレーションのメインループ
while (t := sim.getSimulationTime()) < 300:
    print(f'Simulation time: {t:.2f} [s]')

    # Update Boid positions
    for boid in boids:
        boid.update(boids)

    # Update Kilobot positions
    for i, kilobot in enumerate(kilobots):
        new_position = list(boids[i].position) + [0]
        sim.setObjectPosition(kilobot, -1, new_position)

    sim.step()

# シミュレーションの停止
sim.stopSimulation()

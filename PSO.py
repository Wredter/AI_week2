import random
import math
import numpy as np


def objective_function(x, y):
    value = (1.5 - x - x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
    return value


def calc_move_vector(X, Y):
    move_vector = (X[0] - Y[0], X[1] - Y[1])
    return move_vector


def calc_add_vector(X, Y):
    move_vector = (X[0] + Y[0], X[1] + Y[1])
    return move_vector


def multiply_vector(X, number):
    new_vector = (X[0]*number, X[1]*number)
    return new_vector


class Particle:
    def __init__(self, x, y):
        self.curr_pos = (x, y)
        self.global_best_pos = (0, 0)
        self.global_best_val = np.inf
        self.local_best_pos = (x, y)
        self.local_best_val = objective_function(self.local_best_pos[0], self.local_best_pos[1])
        self.inertia = (random.random()*9-4.5, random.random()*9-4.5)
        self.c1 = 2
        self.c2 = 2
        self.weight = 0.9

    def move(self):
        r1 = random.random()
        r2 = random.random()
        v1 = multiply_vector(self.inertia, self.weight)
        v2 = multiply_vector(multiply_vector(calc_move_vector(self.local_best_pos, self.curr_pos), r1), self.c1)
        v3 = multiply_vector(multiply_vector(calc_move_vector(self.global_best_pos, self.curr_pos), r1), self.c1)
        move_vector = calc_add_vector(calc_add_vector(v2, v3), v1)

        self.curr_pos = calc_add_vector(self.curr_pos, move_vector)
        self.inertia = (random.random() * 9 - 4.5, random.random() * 9 - 4.5)

    def calc_obj(self):
        return objective_function(self.curr_pos[0], self.curr_pos[1])


class PSO:
    def __init__(self):
        self.epochs = 500
        self.number_of_particles = 121
        self.swarm = []
        dist = 9.0 / 12.0
        self.inertia_factor = 0.7 / self.epochs
        for x in range(int(math.sqrt(self.number_of_particles))):
            for y in range(int(math.sqrt(self.number_of_particles))):
                self.swarm.append(Particle(x*dist + dist, y*dist + dist))

    def run(self):
        global_best_val = np.inf
        global_best_pos = (0, 0)
        for epoch in range(self.epochs):
            for particle in self.swarm:
                score = particle.calc_obj()
                if particle.global_best_val > global_best_val:
                    particle.global_best_val = global_best_val
                    particle.global_best_pos = global_best_pos
                if score < global_best_val:
                    global_best_val = score
                    global_best_pos = particle.curr_pos
                if score < particle.local_best_val:
                    particle.local_best_val = score
                    particle.local_best_pos = particle.curr_pos

                particle.move()
                particle.weight = particle.weight - self.inertia_factor
            print(f'Epoka {epoch} z {self.epochs}')
        return global_best_pos, global_best_val



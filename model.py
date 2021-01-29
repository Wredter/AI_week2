import pandas as pd
import numpy as np
import random


class Job:
    def __init__(self, tag):
        self.tag = tag
        self.resource = {}
        self.time = {}
        self.doing = 0
        self.completed = 0  # Pamiętać żeby przy ustawieniu compleated sprawdzić czy nie wykracza za listę i jeśli tak oznaczyć cały job jako zrobiony
        self.time_remaining = 0

    def set_time_rem(self, index):
        if index < len(self.time):
            self.time_remaining = self.time[index]
        else:
            self.time_remaining = 0

    def reduce_time_rem(self, time):
        self.time_remaining -= time
        if self.time_remaining <= 0:
            return self.complete_task()

    def get_time(self):
        if self.completed < len(self.time):
            return self.time[self.completed+1]
        else:
            return 0

    def get_res(self):
        if self.completed < len(self.resource):
            return self.resource[self.completed+1]
        else:
            return 0

    def complete_task(self):
        self.completed += 1
        if self.completed >= len(self.resource):
            return True
        else:
            return False


class AllJobs:
    def __init__(self):
        self.jobs = load_jobs()
        self.jobs_done = []
        for _ in self.jobs:
            self.jobs_done.append(False)

    def calc_cost_of_solution(self, solution):
        selected = [x for x, y in zip(self.jobs, solution) if y == True]
        time_cost = 0
        res_cost = 0
        for job in selected:
            time_cost += job.get_time()
            res_cost += job.get_res()
        if len(selected) == 0:
            return 0, 0
        return time_cost/len(selected), res_cost

    def get_remaining_jobs(self):
        return [x for x, y in zip(self.jobs, self.jobs_done) if y != True]

    def do_jobs(self, solution):
        jobs_to_do = self.get_remaining_jobs()
        selected = [x for x, y in zip(jobs_to_do, solution) if y == True]
        max_time = 0
        min_job = None
        for job in selected:
            job.doing += 1
            job.set_time_rem(job.doing)
            print(f'doing: {job.tag}, completed: {job.doing}|{len(job.time)}')
            if job.time_remaining > max_time:
                max_time = job.time_remaining
        for job in selected:
            if job.reduce_time_rem(max_time):
                self.jobs_done[job.tag] = True

    def max_time(self, solution):
        selected = [x for x, y in zip(self.jobs, solution) if y == True]
        max_time = 0
        for job in selected:
            if job.get_time() > max_time:
                max_time = job.get_time()
        return max_time



class Genetic:
    def __init__(self):
        self.total_time = 0
        self.all_jobs = AllJobs()
        self.total_resource = 150
        self.generations = 75
        self.reproduce_point = int(np.floor(len(self.all_jobs.get_remaining_jobs())*0.7))
        self.current_resource = self.total_resource
        self.mutation_chance_percent = 25 # od 0 do 99
        self.population = 300
        self.parent_search_rage = 10
        self.current_best = None
        self.number_of_moved_solutions = 5

    def generate_rand_solution(self):
        solution = []
        for _ in self.all_jobs.get_remaining_jobs():
            solution.append(random.choice([True, False]))
        return solution

    def generate_max_solution(self):
        solution = []
        for _ in self.all_jobs.get_remaining_jobs():
            solution.append(True)
        return solution

    def calc_cost_of_solution(self, solution):
        time_cost, res_cost = self.all_jobs.calc_cost_of_solution(solution)
        if res_cost > self.current_resource or time_cost == 0:
            return np.inf
        return time_cost + (self.current_resource - res_cost)

    def reproduce(self, solution1, solution2):
        solution1_core = solution1[:self.reproduce_point]
        solution2_core = solution2[:self.reproduce_point]
        solution1_gen = solution1[self.reproduce_point:]
        solution2_gen = solution2[self.reproduce_point:]
        return solution1_core + solution2_gen, solution2_core + solution1_gen

    def mutate(self, solution):
        mutuj = True
        while mutuj:
            chance = random.randrange(101)
            if chance <= self.mutation_chance_percent:
                rand_gen = random.randrange(len(solution))
                solution[rand_gen] = not solution[rand_gen]
            else:
                mutuj = False
        return solution

    def generate_population(self):
        population = []
        for x in range(self.population):
            population.append(self.generate_rand_solution())
        return population

    def natural_selection(self, population):
        fitness_scores = []
        old_parents = []
        new_population = []
        for pop in population:
            fitness_scores.append(self.calc_cost_of_solution(pop))
        temp_best = population[fitness_scores.index(min(fitness_scores))]
        if self.calc_cost_of_solution(temp_best) < self.calc_cost_of_solution(self.current_best):
            self.current_best = temp_best
        while len(new_population) < self.population - self.number_of_moved_solutions:
            parents = random.sample(range(len(population)), self.parent_search_rage)
            min2, min1 = np.inf, np.inf
            p1, p2 = None, None
            pn1, pn2 = 0, 0
            for parent in parents:
                if fitness_scores[parent] < min1 and fitness_scores[parent] < min2:
                    p1 = population[parent]
                    min1 = fitness_scores[parent]
                    pn1 = parent
                if min1 < fitness_scores[parent] < min2:
                    p2 = population[parent]
                    min2 = fitness_scores[parent]
                    pn2 = parent
            if p1 is None or p2 is None:
                return None
            child1, child2 = self.reproduce(p1, p2)
            old_parents.append((pn1, pn2))
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        fitness_scores, population = zip(*sorted(zip(fitness_scores, population)))
        moved_pop = population[:self.number_of_moved_solutions]
        new_population.extend(moved_pop)
        return new_population

    def evolve(self):
        first = True
        population = None
        fitness_scores = []
        self.current_best = self.generate_max_solution()
        for generation in range(self.generations):
            if first:
                population = self.generate_population()
                first = False
            population = self.natural_selection(population)
            if population is None:
                break
        if population is not None:
            for pop in population:
                fitness_scores.append(self.calc_cost_of_solution(pop))
            temp_best = population[fitness_scores.index(min(fitness_scores))]
            if self.calc_cost_of_solution(temp_best) < self.calc_cost_of_solution(self.current_best):
                self.current_best = temp_best
        time, res = self.all_jobs.calc_cost_of_solution(self.current_best)
        if res < self.total_resource:
            return self.current_best
        else:
            print("Nie znaleziono rozwizania")
            return None

    def do_jobs(self, solution):
        self.all_jobs.do_jobs(solution)

    def run(self):
        while not all(self.all_jobs.jobs_done):
            solution = self.evolve()
            if solution:
                self.do_jobs(solution)
                time, res_used = self.all_jobs.calc_cost_of_solution(solution)
                self.total_time += self.all_jobs.max_time(solution)
                print(f'_________________NEXT_TASKS____________________________')
        return self.total_time


def load_jobs():
    df = pd.read_excel(r'Data\GA_task.xlsx')
    jobs = []
    for i in range(50):
        job = df.iloc[1:12, 2 * i: 2 * i + 2]
        jobs.append(Job(i))
        for index, row in job.iterrows():
            jobs[i].resource[index] = row.iloc[0]
            jobs[i].time[index] = row.iloc[1]
    return jobs

import random
from time import clock
import math
from Graph import *


def length(solution, graph):
    l = len(graph.vertices[solution[0]].sequence)
    result = l
    for i in range(1, len(solution)):
        result += l - graph.vertices[solution[i - 1]].overlaps(solution[i])

    return result


def num_of_oligonucleotides(solution):
    return len(set(solution))


def find_successor(solution, vertex_index):
    for i in range(len(solution) - 1):
        if vertex_index == solution[i]:
            return solution[i + 1]
    return -1


def subcycle(solution, vertex_index):
    return vertex_index in solution


def check(solution):
    for s in solution:
        if s == -1:
            print('AHTUNG')
    print(len(set(solution)))


class GeneticAlgorithm:

    def __init__(self, file, n, l, population_size):
        self.graph = Graph(file)
        self.n = n
        self.l = l
        self.population = []
        self.population_size = population_size
        self.best_solution = []

    def fitness(self, solution):
        solution_num = len(solution)
        solution_length = self.l
        max_num = 0
        max_substring = []

        if solution_num > 0:
            substring = [solution[0]]
            i = 1
            while i < solution_num - len(substring) + 1:
                j = i
                while j < solution_num and solution_length <= self.n:
                    substring.append(solution[j])
                    solution_length += self.l - self.graph.vertices[solution[j - 1]].overlaps(solution[j])
                    j += 1
                if solution_length > self.n:
                    solution_length -= self.l - self.graph.vertices[substring[-2]].overlaps(substring[-1])
                    substring.pop()
                if num_of_oligonucleotides(substring) > max_num:
                    max_num = num_of_oligonucleotides(substring)
                    max_substring = substring[:]
                solution_length -= self.l - self.graph.vertices[substring[0]].overlaps(substring[1])
                substring = substring[1:]
                i = j - 1
        else:
            max_num = 0
        return max_num

    def fitness1(self, solution):
        if num_of_oligonucleotides(solution) > 0:
            return 1 / length(solution, self.graph)
        return 0

    def initialize_population(self):
        for i in range(int(self.population_size * 1.5)):
            new = self.graph.random_solution()
            self.population.append(new)

    def new_generation_greedy(self):

        probability = 2 ** (self.fitness(self.best_solution) / num_of_oligonucleotides(self.best_solution)) - 1
        probability *= 0.85
        #probability = 0.8
        random.shuffle(self.population)
        offspring = []
        for i in range(int(self.population_size / 2)):
            child = self.greedy_crossover(self.population[i * 2], self.population[i * 2 + 1], probability)
            self.population.append(child)

    def new_generation_pmx(self):
        random.shuffle(self.population)
        offspring = []
        for i in range(int(self.population_size / 2)):
            child = self.pmx_crossover(self.population[i * 2], self.population[i * 2 + 1])
            offspring.append(child)
            self.population.append(child)

    def selection_tournament(self, q):
        selected_population = []
        for i in range(self.population_size):
            rand = random.sample(range(len(self.population)), q)
            max_fitness = 0
            for j in range(q):
                fitness = self.fitness(self.population[rand[j]])
                if fitness > max_fitness:
                    max_fitness = fitness
                    max_solution = self.population[rand[j]]
            selected_population.append(max_solution)
        self.population = selected_population[:]

    def my_selection(self, q, betha = 1):
        selected_population = []
        for i in range(self.population_size):
            rand = random.sample(range(len(self.population)), q)
            e = []
            sum = 0
            for index in rand:
                e_z = math.exp(betha * self.fitness(self.population[index]))
                e.append(e_z)
                sum += e_z
            for j in range(len(e)):
                e[j] = e[j] / sum
            rand_number = random.random()
            previous = 0
            for j in range(len(e)):
                if previous + e[j] >= rand_number:
                    selected_population.append(self.population[rand[j]])
                    break
                previous += e[j]
        self.population = selected_population[:]

    def select_the_best(self):
        max_fitness = 0
        for solution in self.population:
            fitness = self.fitness(solution)
            if fitness > max_fitness:
                max_fitness = fitness
                best = solution[:]
        if max_fitness > self.fitness(self.best_solution):
            self.best_solution = best[:]

    def greedy_crossover(self, parent1, parent2, probability):
        first = parent1[random.randint(0, len(parent1) - 1)]
        child = [first]
        for i in range(len(parent1) - 1):

            rand = random.random()

            if rand < probability:
                successor1 = find_successor(parent1, child[-1])
                successor2 = find_successor(parent2, child[-1])

                successor = -1
                if self.graph.vertices[child[-1]].overlaps(successor1) >= self.graph.vertices[child[-1]].overlaps(successor2):
                    if not subcycle(child, successor1) and successor1 != -1:
                        successor = successor1
                else:
                    if not subcycle(child, successor2) and successor2 != -1:
                        successor = successor2
                if successor == -1:
                    rand = [i for i in range(len(parent1))]
                    random.shuffle(rand)
                    for vertex_index in rand:
                        if not subcycle(child, vertex_index):
                            successor = vertex_index
                            break
            else:
                max_overlap = 0
                for potential in range(self.graph.numOfVertices):
                    if not subcycle(child, potential):
                        if self.graph.vertices[child[-1]].overlaps(potential) >= max_overlap:
                            max_overlap = self.graph.vertices[child[-1]].overlaps(potential)
                            successor = potential
                        if max_overlap == self.l - 1:
                            break

            child.append(successor)
        return child

    def pmx_crossover(self, parent1, parent2):
        segment_length = 10
        start = random.randint(0, len(parent1) - segment_length)
        #start = 3
        child = parent2[:]
        segment = parent1[start:start + segment_length]
        segment_in_parent2 = parent2[start:start + segment_length]
        child[start:start + segment_length] = segment[:]
        #print(segment)
        #print(child)
        for i in range(segment_length):
            if segment_in_parent2[i] not in segment:
                #print('BRAKUJE', segment_in_parent2[i], i)
                #print(segment[i])
                inserted = False
                to_replace = segment[i]
                while not inserted:
                    for j in range(len(parent2)):
                        if parent2[j] == to_replace:
                            if j < start or j >= start + segment_length:
                                child[j] = segment_in_parent2[i]
                                inserted = True
                            else:
                                to_replace = parent1[j]
                                #print(to_replace)

        #print(child)
        return child

    def inverse_mutation(self, solution_index, segment_length, probability):
        rand = random.random()
        if rand < probability / 100:
            start = random.randint(0, len(self.population[solution_index]) - segment_length)
            segment = self.population[solution_index][start:start + segment_length]
            segment.reverse()
            self.population[solution_index][start:start + segment_length] = segment[:]

    def scramble_mutation(self, solution_index, segment_length, probability):
        rand = random.sample(range(len(self.population[solution_index])), segment_length)
        segment = []

        for i in rand:
            segment.append(self.population[solution_index][i])

        random.shuffle(segment)

        for i in range(segment_length):
            self.population[solution_index][rand[i]] = segment[i]

    def make_mutations(self, segment_length, probability):
        for i in range(len(self.population)):
            if random.random() < 0.5:
                self.scramble_mutation(i, segment_length, probability)
            else:
                self.inverse_mutation(i, segment_length, probability)

    def add_best_to_population(self):
        best = self.best_solution[:]
        self.population.append(best)
        print(self.fitness(self.best_solution), self.fitness1(self.best_solution), length(self.best_solution, self.graph))

    def calculate_new_length(self, length, solution, position1, position2):
        vertex1 = self.graph.vertices[solution[position1]]
        vertex2 = self.graph.vertices[solution[position2]]

        if position1 > 0:
            length -= self.l - self.graph.vertices[solution[position1 - 1]].overlaps(vertex1.index)
            length += self.l - self.graph.vertices[solution[position1 - 1]].overlaps(vertex2.index)
            length -= self.l - vertex1.overlaps(solution[position1 + 1])
            length += self.l - vertex2.overlaps(solution[position1 + 1])
        else:
            length -= self.l - vertex1.overlaps(self.graph.vertices[solution[position1 + 1]])
            length += self.l - vertex2.overlaps(self.graph.vertices[solution[position1 + 1]])
        if position2 < len(solution) - 1:
            length -= self.l - self.graph.vertices[solution[position2 - 1]].overlaps(vertex2.index)
            length += self.l - self.graph.vertices[solution[position2 - 1]].overlaps(vertex1.index)
            length -= self.l - vertex2.overlaps(solution[position2 + 1])
            length += self.l - vertex1.overlaps(solution[position2 + 1])
        else:
            length -= self.l - self.graph.vertices[solution[position2 - 1]].overlaps(vertex2.index)
            length += self.l - self.graph.vertices[solution[position2 - 1]].overlaps(vertex1.index)

        return length

    def two_opt(self, solution):
        repeat = True
        count = 0
        while repeat:
            max_fitness = 0
            solution_length = length(solution, self.graph)
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    new_solution = solution[:]
                    tmp = new_solution[i]
                    new_solution[i] = new_solution[j]
                    new_solution[j] = tmp
                    fitness = 1 / self.calculate_new_length(solution_length, solution, i, j)
                    if 1 / solution_length < fitness:
                        if fitness > max_fitness:
                            max_fitness = fitness
                            max_solution = new_solution[:]
            if max_fitness > 0:
                solution = max_solution[:]
            else:
                repeat = False
            count += 1
            if count >= 3:
                repeat = False

        return solution

    def genetic_algorithm(self, generations):
        self.initialize_population()
        for i in range(generations):
            self.select_the_best()
            self.my_selection(3, betha=1)
            self.add_best_to_population()
            self.new_generation_greedy()
            self.make_mutations(3, 5)

    def make_measurements(self):
        optimum = 440
        start = clock()
        previous_best = 0
        self.initialize_population()
        for i in range(50):
            self.select_the_best()
            if self.fitness(self.best_solution) == optimum:
                return self.fitness(self.best_solution), clock() - start
            if previous_best < self.fitness(self.best_solution):
                achieved_time = clock() - start
                previous_best = self.fitness(self.best_solution)
            #print('Achieved time', achieved_time)
            self.my_selection(3, betha=1)
            self.add_best_to_population()
            self.new_generation_greedy()
            self.make_mutations(3, 5)
        return self.fitness(self.best_solution), achieved_time

ga = GeneticAlgorithm('Instances/positiveErrorsDistortions/instance4', 309, 10, 100)
#ga.graph.greedy_algorithm(209, 10)
ga.genetic_algorithm(100)

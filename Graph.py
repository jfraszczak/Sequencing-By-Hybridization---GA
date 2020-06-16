import random


def sequences_overlap(sequence1, sequence2):
    n = len(sequence1)
    for i in range(1, n):
        if sequence1[i:] == sequence2[:n - i]:
            return n - i
    return 0


class Vertex:
    def __init__(self, sequence, index):
        self.sequence = sequence
        self.neighbours = []
        self.index = index

    def add_neighbour(self, neighbour_index, overlap):
        neighbour = {'index': neighbour_index, 'overlap': overlap}
        self.neighbours.append(neighbour)

    def show_vertex(self):
        print(self.index, self.sequence)
        print(self.neighbours)

    def is_neighbour(self, vertex_index):
        for neighbour in self.neighbours:
            if vertex_index == neighbour['index']:
                return True
        return False

    def overlaps(self, vertex_index):
        for neighbour in self.neighbours:
            if neighbour['index'] == vertex_index:
                return neighbour['overlap']
        return 0


class Graph:
    def __init__(self, file):
        self.vertices = []
        self.numOfVertices = 0
        self.read_file(file)
        self.set_neighbours()

    def read_file(self, file_name):
        file = open(file_name, 'r')
        for sequence in file:
            if sequence[-1] == '\n':
                sequence = sequence[:-1]
            vertex = Vertex(sequence, self.numOfVertices)
            self.vertices.append(vertex)
            self.numOfVertices += 1

    def set_neighbours(self):
        for i in range(self.numOfVertices):
            k = 0
            for j in range(self.numOfVertices):
                if i != j:
                    overlap = sequences_overlap(self.vertices[i].sequence, self.vertices[j].sequence)
                    self.vertices[i].add_neighbour(j, overlap)
                    if overlap == 0:
                        k += 1
            #print(k)

    def greedy_algorithm(self, n, l):
        max_s = 0
        for i in range(self.numOfVertices):
            s = 1
            length = l
            vertices = [i]
            repeat = True
            while repeat:
                cost = l
                for neighbour in self.vertices[vertices[-1]].neighbours:
                    if l - neighbour['overlap'] <= cost and neighbour['index'] not in vertices:
                        cost = l - neighbour['overlap']
                        index = neighbour['index']
                        if cost == 1:
                            break
                if length + cost <= n:
                    length += cost
                    s += 1
                    vertices.append(index)
                else:
                    repeat = False

            if s > max_s:
                max_s = s
                max_length = length
                max_vertices = vertices[:]

        print('GREEDY SOLUTION')
        print(max_s, max_length)
        return max_vertices

    def random_solution(self):
        new_solution = [i for i in range(self.numOfVertices)]
        random.shuffle(new_solution)
        return new_solution

    def greedy_solution(self, n, l, quantity_of_spectrum):
        solution = self.greedy_algorithm(n, l)
        rand = [i for i in range(quantity_of_spectrum)]
        random.shuffle(rand)
        for i in rand:
            if i not in solution:
                solution.append(i)

        return solution

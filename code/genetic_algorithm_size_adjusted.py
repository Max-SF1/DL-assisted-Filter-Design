from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import heapq
import cnn_model
import seed_generator as sg
import math

# CNN INITIALIZATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_model = cnn_model.cnn_model(output_w=12)
loaded_model.load_state_dict(torch.load('final_cnn_model.pth'))
loaded_model.to(device)
loaded_model.eval()

# dB Function
def dB(Re, Im):
    h = math.sqrt(Re ** 2 + Im ** 2)
    return 20 * math.log(h, 10)

# GENETIC ALGORITHM CLASS

def signed_distance_lower(s_param, constraint):
    S_score = 0
    for i in range(len(s_param)):
        if s_param[i] > constraint:
            S_score += abs(s_param[i] - constraint)
        # else:
        #     S_score -= abs(s_param[i] - constraint)
    return -S_score

def signed_distance_higher(s_param, constraint):
    S_score = 0
    for i in range(len(s_param)):
        if s_param[i] < constraint:
            S_score += abs(s_param[i] - constraint)
        # else:
        #     S_score -= abs(s_param[i] - constraint)
    return -S_score

class Member:
    def __init__(self, seed, model, device):
        self.seed = seed
        self.rows = seed.shape[0]
        self.columns = seed.shape[1]
        self._score = None
        self.model = model
        self.device = device

    def score(self):
        if self._score is None:
            seed_tensor = torch.FloatTensor(self.seed).unsqueeze(0).unsqueeze(0).to(self.device)
            output_matrix = self.model(seed_tensor).cpu().detach().numpy()
            S_11_low = output_matrix[0, 0, 0:5]
            S_21_low = output_matrix[0, 1, 0:5]
            S_21_high = output_matrix[0, 1, 6:]
            # S_22_low = output_matrix[0, 2, 0:5]
            s_11_score = signed_distance_lower(S_11_low, -15)
            s_21_score_1 = signed_distance_higher(S_21_low, -0.5)
            s_21_score_2 = signed_distance_lower(S_21_high, -15)
            self._score = s_11_score + s_21_score_1 + s_21_score_2
        return self._score

    def mutate(self, mutation_rate):
        self.seed = sg.seed_mutation(self.seed, mutation_rate)
        self._score = None

def create_offsprings(member_1: Member, member_2: Member) -> List[Member]:
    center_1, l_1, r_1 = sg.strip_seed(member_1.seed)
    center_2, l_2, r_2 = sg.strip_seed(member_2.seed)
    new_seed = np.zeros(center_1.shape)
    for i in range(center_1.shape[0]):
        j = np.random.randint(1, center_1.shape[1] - 2)
        new_seed[i] = np.append(center_1[i][:j], center_2[i][j:])
    random_number_1 = np.random.randint(2)
    random_number_2 = np.random.randint(2)
    l_1_prime = l_1 if random_number_1 else l_2
    r_1_prime = r_1 if random_number_2 else r_2
    return [Member(sg.stack_seed(new_seed, l_1_prime, r_1_prime), member_1.model, member_1.device)]

class Population:
    mutation_rate: float

    def __init__(self, population_size, rows, columns, device):
        self.fitness = None
        self.generation_count = 0
        self.population_size = population_size
        self.rows = rows
        self.columns = columns
        self.mutation_rate = 0.1
        self.device = device
        self.model = cnn_model.cnn_model(output_w=12).to(device)
        self.model.load_state_dict(torch.load('final_cnn_model.pth'))
        self.model.eval()
        self.members = [Member(sg.random_seed(rows, columns), self.model, self.device) for _ in range(population_size)]
        self.update_population_fitness()

    def update_population_fitness(self):
        total_fitness = sum(member.score() for member in self.members)
        self.fitness = [member.score() / total_fitness for member in self.members]

    def print_best_seed(self):
        best_member = max(self.members, key=lambda x: x.score())
        return best_member

    def choose_parents(self, num_of_parents):
        heap = [(member.score(), id(member), member) for member in self.members[:num_of_parents]]
        heapq.heapify(heap)
        for member in self.members[num_of_parents:]:
            score = member.score()
            if score > heap[0][0]:
                heapq.heappushpop(heap, (score, id(member), member))
        parents = [member for _, _, member in heap]
        return parents

    def mutate(self, member_list):
        for member_inst in member_list:
            member_inst.mutate(mutation_rate=self.mutation_rate)
        return member_list

    def tournament_selection(self, tournament_size, offsprings_num):
        new_pop = []
        for _ in range(offsprings_num):
            competitors = random.sample(self.members, tournament_size)
            sorted_competitors = sorted(competitors, key=lambda x: x.score(), reverse=True)
            parent1, parent2 = sorted_competitors[0], sorted_competitors[1]
            new_pop.extend(create_offsprings(parent1, parent2))
        new_pop = self.mutate(new_pop)
        return new_pop

    def new_generation(self):
        crossover_num = 8
        tournament_size = 400
        random_num = 50
        crossovers = self.choose_parents(crossover_num)
        offsprings_num = self.population_size - crossover_num - random_num
        new_pop = self.tournament_selection(tournament_size=tournament_size, offsprings_num=offsprings_num)
        new_pop.extend(crossovers)
        for _ in range(random_num):
            new_pop.append(Member(sg.random_seed(self.rows, self.columns), self.model, self.device))
        self.members = new_pop
        self.update_population_fitness()
        self.mutation_rate = max(0, self.mutation_rate - 0.001)
        print(f"Just finished generation {self.generation_count}!")
        self.generation_count += 1

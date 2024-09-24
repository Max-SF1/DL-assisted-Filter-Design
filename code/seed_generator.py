import numpy as np
import matplotlib.pyplot as plt
import random


def random_seed(rows, cols):  # generates a random seed with random port locations.
    l_port_location = random.randint(0, rows - 1)
    r_port_location = random.randint(0, rows - 1)
    l_vect = np.zeros(rows)
    l_vect[l_port_location] = 1
    r_vect = np.zeros(rows)
    r_vect[r_port_location] = 1
    middle = np.random.randint(2, size=(rows, cols - 2))
    seed = np.c_[l_vect, middle, r_vect]
    return (seed)

def stack_seed(center,l_port_location, r_port_location,rows=16, cols=18):
    l_vect = np.zeros(rows)
    l_vect[l_port_location] = 1
    r_vect = np.zeros(rows)
    r_vect[r_port_location] = 1
    seed = np.c_[l_vect, center, r_vect]
    return (seed)

def strip_seed(member): #returns the matrix and two port locations
    rows, cols = member.shape
    idx_l= 0
    idx_r = 0
    center = member[:, 1:cols - 1]
    l_vect = member[:,0]
    r_vect = member[:, cols  -1 ]
    # print(f'l_vect: {l_vect}')
    # print(f'r_vect: {r_vect}')
    for i in range(len(l_vect)):
        if l_vect[i] == 1:
            idx_l = i
            break
    for j in range(len(r_vect)):
        if r_vect[j] == 1:
            idx_r = j
            break
    return center, idx_l, idx_r

def seed_mutation(seed, mutation_rate):
    rows, cols = seed.shape
    center = seed[:, 1:cols - 1]  # only mutate the random seed in the middle, not the first and last cols

    # Apply mutation to center based on mutation_rate
    mask = np.random.uniform(0, 1, size=center.shape) < mutation_rate
    m_cent = np.where(mask, 1 - center, center)

    # Mutate left and right ports
    l_vect = seed[:, 0]  # get the first column
    r_vect = seed[:, cols - 1]  # get the last column
    if np.random.uniform(0, 1) <= mutation_rate:  # port mutation
        l_vect = np.roll(l_vect, np.random.randint(-2, 2))  # randomly choose cyclic shift amount
    if np.random.uniform(0, 1) <= mutation_rate:  # port mutation
        r_vect = np.roll(r_vect, np.random.randint(-2, 2))  # randomly choose cyclic shift amount for port.

    # Concatenate the mutated parts to form the mutated seed
    mutated_seed = np.concatenate((l_vect.reshape(-1, 1), m_cent, r_vect.reshape(-1, 1)), axis=1)
    return mutated_seed


import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from cnn_model import cnn_model_1_conv
import seed_generator as sg
import simplified_cnn_model
import math
import genetic_algorithm_size_adjusted as ga

def generate_matrix(grid):
    grid_str = ""
    for row_idx, row in enumerate(grid):
        grid_str += " ".join(map(str, row))
        if row_idx != len(grid) - 1:
            grid_str += " "
        grid_str += "\n"
    return grid_str

generations = 100
saved_data = torch.load('preprocessed_data_16.pth')
output_w = saved_data['output_w']
dataset = saved_data['test_dataset']
plot_freqs = saved_data['plot_freqs']

# CNN INITIALIZATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_model = cnn_model_1_conv(output_w=12)
loaded_model.load_state_dict(torch.load('FR_1conv_truly_cnn_model_1_conv.pth'))
loaded_model.to(device)
loaded_model.eval()

# Initialize genetic algorithm parameters
rows = 16
columns = 18
population_size = 4000

# Training loop
pop = ga.Population(population_size=population_size, rows=rows, columns=columns, device=device)
best_scores = []

for generation in range(generations):
    pop.new_generation()
    print(generation)
    if (generation == 99) or (generation == 100):
        best_member = pop.print_best_seed()
        best_image = best_member.seed
        best_scores.append(best_member.score())

        print(best_image)
        print(generate_matrix(best_image))

        new_data = torch.FloatTensor(best_image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output_matrix = loaded_model(new_data).cpu().numpy()

        dB_S11 = output_matrix[0, 0,0, :]
        dB_S21 = output_matrix[0, 0,1, :]
        dB_S22 = output_matrix[0, 0,2, :]
        plt.figure(figsize=(10, 6))
        plt.plot(plot_freqs, dB_S11, 'o-', color='b', label='Predicted $S_{11}$ dB')
        plt.plot(plot_freqs, dB_S21, 'o-', color='r', label='Predicted $S_{21}$ dB')
        plt.plot(plot_freqs, dB_S22, 'o-', color='g', label='Predicted $S_{22}$ dB')
        plt.title(f'Predicted Filter Performance - Generation {generation}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.grid(True)
        plt.show()

plt.figure(figsize=(10, 6))
plt.plot(best_scores, 'o-', color='b', label='Best Score')
plt.title('Best Score as a Function of Generation')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.legend()
plt.grid(True)
plt.show()

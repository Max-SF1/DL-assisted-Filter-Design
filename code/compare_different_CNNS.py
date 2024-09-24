import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cnn_model import skip_and_pool, cnn_model_short

saved_data = torch.load('preprocessed_data_16.pth')
train_dataset = saved_data['train_dataset']
validation_dataset = saved_data['validation_dataset']
output_w = saved_data['output_w']
run_name = input("Please enter the run name: ")

db = []
models = [
skip_and_pool,cnn_model_short, ]
torch.cuda.empty_cache()

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def train_loop_with_lr_reduction(model, train_loader, validation_loader, optimizer, criterion, device, num_epochs=150):
    global early_stopping_counter
    start_time = time.time()
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    validation_losses = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            # You can disable CuDNN for specific forward pass if needed
            with torch.backends.cudnn.flags(enabled=False):
                outputs = model(data)

            targets = targets.view_as(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, targets in validation_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                targets = targets.view_as(outputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_loader)
        validation_losses.append(avg_val_loss)
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{run_name}_{model.__class__.__name__}.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 8:
                print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                break

        print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Training completed in {total_time:.2f} seconds. Best validation loss: {best_val_loss:.4f}')
    return best_val_loss, train_losses, validation_losses, total_time


results = []

results_directory = "training_results"
os.makedirs(results_directory, exist_ok=True)


def save_results_to_json(model_name, results, directory="training_results"):
    filepath = os.path.join(directory, f"{run_name}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            all_results = json.load(file)
        all_results[model_name] = results  # Update existing dictionary with new results
    else:
        all_results = {model_name: results}
    with open(filepath, "w") as file:
        json.dump(all_results, file, indent=4)


for i in range(len(models)):
    model = models[i](output_w=output_w).to(device)
    model_name = model.__class__.__name__
    print(f'Currently training: {model_name}')
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.L1Loss()

    best_val_loss, train_losses, validation_losses, train_time = train_loop_with_lr_reduction(
        model, train_loader, validation_loader, optimizer, criterion, device, num_epochs=150
    )

    results = {
        'model': model_name,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'validation_losses': validation_losses,
        'train_time': train_time
    }

    # Save results for the current model
    save_results_to_json(model_name, results)


def plot_results_from_json(filepath=f"training_results/{run_name}.json"):
    with open(filepath, "r") as json_file:
        all_results = json.load(json_file)
    plt.figure(figsize=(12, 8), facecolor='black')
    styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    with plt.style.context('dark_background'):
        for idx, (model_name, results) in enumerate(all_results.items()):
            style_idx = idx % len(styles)
            color_idx = idx % len(colors)
            train_style = colors[color_idx] + styles[style_idx]
            val_style = colors[color_idx] + styles[(style_idx + 1) % len(styles)]

            plt.plot(results['validation_losses'], val_style, linewidth=4,
                     label=f"{model_name} Validation Loss (Thicker)")
            plt.plot(results['train_losses'], train_style, linewidth=2, label=f"{model_name} Training Loss")

        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.title('Validation Loss Across Models', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.75)
        ax = plt.gca()
        ax.set_facecolor('#112233')
        plt.show()


plot_results_from_json(f"training_results/{run_name}.json")

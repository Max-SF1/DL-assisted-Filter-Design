import ast
import pandas as pd
import numpy as np
import torch
from CustomDataset import CustomDataset
from sklearn.model_selection import train_test_split


def dB_np(x_re, x_im):
    return 20 * np.log10(np.sqrt(x_re ** 2 + x_im ** 2))  # Removed the previously Added epsilon for numerical stability


def load_and_combine_csv_files(filenames):
    df_list = [pd.read_csv(filename) for filename in filenames]
    combined_df = pd.concat(df_list, ignore_index=True)
    for column in ['s11_re', 's11_im', 's21_re', 's21_im', 's22_re', 's22_im']:
        combined_df[column] = combined_df[column].apply(
            lambda x: np.array(ast.literal_eval(x.replace('{', '[').replace('}', ']'))))
    return combined_df


def preprocess_qr_and_ports(qr_string, ports_string):
    ports = list(map(int, ports_string.split()))
    port1, port2 = ports[0] - 1, ports[1] - 1
    # Filter out rows that are not strictly binary before converting
    qr_array = np.array(
        [[int(bit) for bit in row.strip() if bit in '01'] for row in qr_string.split('\n') if row.strip()],
        dtype=np.float32)
    l_vect = np.zeros((qr_array.shape[0], 1))
    r_vect = np.zeros((qr_array.shape[0], 1))
    l_vect[port1, 0] = 1
    r_vect[port2, 0] = 1
    return np.hstack([l_vect, qr_array, r_vect])


def create_augmented_dataset(df, indices):
    augmented_data = []
    for _, row in df.iterrows():
        qr_processed = preprocess_qr_and_ports(row['qr'], row['ports'])
        s_params = {
            's11_re': row['s11_re'][indices],
            's11_im': row['s11_im'][indices],
            's21_re': row['s21_re'][indices],
            's21_im': row['s21_im'][indices],
            's22_re': row['s22_re'][indices],
            's22_im': row['s22_im'][indices]
        }

        # Convert S-parameters to dB
        dB_s11 = dB_np(s_params['s11_re'], s_params['s11_im'])
        dB_s21 = dB_np(s_params['s21_re'], s_params['s21_im'])
        dB_s22 = dB_np(s_params['s22_re'], s_params['s22_im'])

        # Combine into a single matrix with dB S-parameters as rows
        s_parameters_matrix = np.stack([dB_s11, dB_s21, dB_s22])

        augmented_data.append((qr_processed, s_parameters_matrix))
    return augmented_data


filenames = [
    "measurements_16x16_1.csv",
    "measurements_16x16_2.csv",
    "measurements_16x16_3.csv",
    "measurements_16x16_4.csv",
    "measurements_16x16_5.csv",
    "measurements_16x16_6.csv",
    "measurements_16x16_7.csv",
    "measurements_16x16_8.csv",
    "measurements_16x16_9_4146 semples.csv",
    "measurements_16x16_10.csv",
    "measurements_16x16_11.csv",
    "measurements_16x16_12_4343 semples.csv",
    "measurements_16x16_13_2320  samples.csv",
    "measurements_16x16_14.csv",
    "measurements_16x16_16_1374 samples.csv",
    "measurements_16x16_18_ 3094 samples.csv",
    "measurements_16x16_17.csv",
    "measurements_16x16_19.csv",
    "measurements_16x16_24.csv",
    "measurements_16x16_21.csv",
    "measurements_16x16_26.csv",
    "measurements_16x16_20.csv",
    "measurements_16x16_22.csv",
    "measurements_16x16_28.csv",
    "measurements_16x16_30.csv",
    "measurements_16x16_23_175 samples.csv",
    "measurements_16x16_25.csv",
    "measurements_16x16_15.csv",
    "measurements_16x16_32_6185 samples.csv"

]

# for the 8x8:
# filenames = ['measurements_8x8_0.csv', 'measurements_8x8_1.csv', 'measurements_8x8_2.csv', 'measurements_8x8_3.csv',
#            'measurements_8x8_4.csv', 'measurements_8x8_5.csv', 'measurements_8x8_6.csv', 'measurements_8x8_7.csv',
#           'measurements_8x8_8.csv', 'measurements_8x8_9.csv', 'measurements_8x8_10.csv', 'measurements_8x8_11.csv',
#          'measurements_8x8_12.csv']
combined_df = load_and_combine_csv_files(filenames)
print(f"Number of samples in combined DataFrame: {len(combined_df)}")

# start, end, pts = 74, 174, 100
indices =  [100,101,102, 120, 121, 122, 130, 131, 132, 138, 139,140 ] #np.linspace(start, end, pts, dtype=int).astype(int)
freqs_string = combined_df.loc[0, 'freq']
freqs_list = ast.literal_eval(freqs_string.replace('{', '[').replace('}', ']'))
plot_freqs = np.array(freqs_list)[indices]
augmented_data = create_augmented_dataset(combined_df, indices)
print(f"Number of samples in augmented data: {len(augmented_data)}")

# print(plot_freqs)
# for i in range(len(plot_freqs)):
#     print(f'freqs_string[{i}] = {plot_freqs[i] / 1e9} Ghz')
# Splitting dataset

train_data, test_data = train_test_split(augmented_data, test_size=0.1, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)
test_dataset = CustomDataset(test_data)
# After preprocessing and extracting plot_freqs
print(len(train_dataset), len(test_dataset), len(val_dataset))
torch.save({
    'train_dataset': train_dataset,
    'validation_dataset': val_dataset,
    'test_dataset': test_dataset,
    'output_w': len(indices),
    'plot_freqs': plot_freqs,
    # Additional information like selected frequencies # for the 70 points, the frequencies in the indices
}, 'preprocessed_data_16.pth')

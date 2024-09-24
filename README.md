# Deep-Learning-assisted-RF-Filter-Design
Generating an Electrical Filter Design in a pixelated design space with a genetic algorithm, and rapid evaluation via a CNN.
This project explores machine learning for the design of Radio Frequency (RF) filters. By leveraging convolutional neural networks (CNNs) and genetic algorithms (GAs), we aim to design RF filters with specific scattering parameter requirements. The design space is a pixelated metal sheet over a substrate, that via intricate selection of nontrivial arbitrary-seeming pixel patterns and the complex EM interactions between them we're able to satisfy filter requirements. No analytical model for the design space exists. 
## Why Use Machine Learning for RF Filter Design?
Conventional RF filter design techniques involve manually selecting configurations and parameter sweeps, which are time-consuming and limited by human expertise. Machine learning expands the design space significantly by automating and optimizing design generation. Also, I'm personally more interested in Machine Learning and wanted to try my hand at coding a Genetic Algorithm.
## Approach
The filter-fabrication pipeline is comprised of several programmatic components, that are built on top of each other.
first, we build a dataset of arbitrary Electromagnetic Structures inside AWR with an automated script that was coded in python using an API, and visual basic, inside of AWR itself. The script generates sets of samples, due to theamount of samples and complexity of EM simulation the set generation requires an immense amount of computation and generation time.
We export the sets to .csv files, which we subsequently open in Python. Each .csv file contains 10, 000 samples (unless a run was interrupted) and their scattering parameters over 195 frequency points.
Inside Python, we perform prepossessing - both changing the way the data is presented to be compatible with Python's numpy and PyTorch libraries, and using the passivity of the scattering-parameters to triple the amount of training and validation samples. Our code lets us select discrete frequency points, modify the CNN and optimization process to revolve solely around them (due to high compute requirements).
<p align="center">
  <img src="assets/symmetry.png" alt="using passivity and symmetry in pixelated RF structures to triple the dataset" width="800"/>
</p>
<p style="text-align:center; font-size: 18px;"><strong>Figure 1: using passivity and symmetry in pixelated RF structures to triple the dataset, flipped structures have similar/transformed scattering parameters</strong></p>
The data, now saved as a PyTorch (A Python machine learning library) dataset, is used to train a convolutional neural network to estimate the frequency response parameters of the structures.
The CNN's accuracy was improved with the use of point-wise convolutions, a design element not previously present in any of the articles.
The weights of the trained model are saved, and the model is loaded into a Genetic Algorithm which optimizes designs based on their predicted frequency response by the model.
The selection process is done via tournament selection based on the cost function, crossover, and mutation - which contain unique considerations for port location and mutation.
The iteration process is repeated for N generations, afterwards the best filter according to the CNN and the cost function is presented to the user. 
Optimization time is greatly minimized via the use of Min-Heaps to select the k-best designs. 
<p align="center">
  <img src="assets/pipeline.png" alt="The pipeline figure" width="800"/>
  
  <strong> Figure 2: the overall design pipeline, a genetic algorithm based on a CNN</strong>
</p>

For an in-depth overview of every pipeline element as well as design process, please look at the Project Report.

## Architecture

1. **Data Generation**: Using AWR and Python scripts, a large dataset of 450,000 structures is generated and their scattering parameters simulated. These structures are represented as 16x16 binary matrices. 
   
2. **CNN for Parameter Prediction**: A CNN is trained to predict the scattering parameters of these structures. The architecture consists of convolutional interspersed with Batch-Norm layers, followed by a point-wise convolution into fully connected layers, achieving a 0.38 dB MAE on test data. Consistently, the addition of data has yielded vastly better performance - we believe more data will lead to a viable pipeline.

3. **Genetic Algorithm for Optimization**: The GA evolves filter designs by iteratively selecting and mutating structures. It uses a score function based on how well the predicted scattering parameters meet design requirements.

## Files 

| File                           | Description                                                                                          |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| **dataset_generation.py**       | Preprocesses the AWR generated files into a PyTorch dataset, performs the symmetry flip                                   |
| **cnn_model.py**                | Defines the CNN architecture used for predicting scattering parameters.                              |
| **genetic_algorithm.py**        | Implements the GA that optimizes RF filter designs.                                                   |
| **dataset_preprocessing.py**    | Preprocesses the generated dataset for training.                                                      |
| **train_model.py**              | Trains the CNN using the preprocessed dataset.                                                        |
| **main.py**                     | Runs the Genetic Algorithm                           |

This Repo was written by Aylon Feraru and contains his contributions to the project, to get the dataset simulation files contact Eric Green or Dan Fishler. 
## Results

- **Accuracy**: Our CNN achieved a Mean Absolute Error (MAE) of 0.38 dB on the prediction of scattering parameters.
- **Optimization**: The GA successfully optimized filter designs, approaching the desired RF characteristics.
  
Performance can be further improved with a larger dataset and extended training.


## References

1. E. A. Karahan et al., “Deep-learning-based inverse-designed millimeter-wave passives and power amplifiers,” *IEEE Journal of Solid-State Circuits*, 2023.
2. A. Gupta et al., “Tandem neural network based design of multiband antennas,” *IEEE Transactions on Antennas and Propagation*, 2023.
3. Y. Li et al., “Predicting scattering from complex nano-structures via deep learning,” *IEEE Access*, 2020.

# FedLearning
Federated Learning Task with Straggler Simulation  This repository implements federated learning algorithms (FedAvg, FedProx, q-FedAvg) to evaluate their performance under different data distributions (IID vs Non-IID) and straggler percentages (0%, 40%, 80%). This simulates real-world scenarios over multiple rounds (20 rounds, 10 local epochs).
# Federated Learning Algorithms Evaluation

This repository implements and evaluates three federated learning (FL) algorithms: **FedAvg**, **FedProx**, and **q-FedAvg**, across multiple scenarios to analyze their performance under various conditions.

## Task Overview

The main objective of this project is to evaluate the performance of different FL algorithms under the following conditions:

- **Data Distribution**: IID (Independent and Identically Distributed) and Non-IID (non-identical distributions across clients).
- **Straggler Simulation**: Varying percentages of straggler clients (0%, 40%, and 80%) to simulate client dropout in each round.
  
The evaluation is done over 20 communication rounds with 10 local epochs per round, with client fractions and local training epochs optimized for efficient testing.

## Algorithms Implemented

### 1. **FedAvg**:
   - The standard federated averaging algorithm.
   - Clients locally compute gradients, which are averaged to update the global model.

### 2. **FedProx**:
   - An extension of FedAvg that adds a proximal term to handle data heterogeneity, controlled by the hyperparameter `mu`.

### 3. **q-FedAvg**:
   - A weighted version of FedAvg that accounts for fairness among clients by introducing a parameter `q`.

## Experiment Setup

- **Data Distribution**: Synthetic datasets with both IID and Non-IID splits are used.
- **Metrics**: The main performance metrics include training loss and validation accuracy.
- **Straggler Rates**: The performance is evaluated with 0%, 40%, and 80% of clients dropping out in each communication round.

## Results and Analysis

- **Performance**: The algorithms' loss and accuracy are visualized across different rounds.
- **Straggler Behavior**: Straggler rates significantly affect the algorithms' convergence, with FedAvg performing better under lower straggler conditions.
- **Data Heterogeneity**: FedProx and q-FedAvg showed better robustness with Non-IID data compared to FedAvg.



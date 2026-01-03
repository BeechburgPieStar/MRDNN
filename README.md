# MRDNN

This repository provides the official implementation of **MRDNN**, a bio inspired multi resolution dynamic neural network for nonlinear behavioral modeling of wideband power amplifiers.

The code corresponds to the paper:

**Power Amplifier Behavioral Modeling Using Bio Inspired Multi Resolution Dynamic Neural Network**

All source code, trained model weights, and training logs are publicly available in this repository.

---

## 1. Overview

Power amplifiers in wideband wireless communication systems exhibit strong nonlinearity and memory effects, leading to spectral regrowth and adjacent channel interference. Although deep learning based behavioral models have achieved promising accuracy, their robustness and adaptability under dynamic signal conditions remain challenging.

MRDNN addresses these issues by integrating:

- A multi resolution interactive fusion module to jointly capture short term nonlinearities and long term memory effects
- A bio inspired dynamic modeling module based on liquid neural networks to adapt to time varying signal environments

The proposed framework focuses not only on time domain modeling accuracy but also on adjacent channel spectral suppression.

---

## 2. Repository Structure

The main structure of this repository is as follows:

```
MRDNN/
├── Datasets/ # Dataset related files
├── Model/ # MRDNN model definitions
├── data_save/ # Saved intermediate data and outputs
├── logs/ # Training logs
├── model_save/ # Trained model weights
├── main.py # Training entry script
├── Evaluation.py # Evaluation script
├── Parameter.py # Hyperparameter configuration
├── data_processing.py # Data preprocessing
├── load_dataset.py # Dataset loading utilities
├── flops.py # FLOPs and complexity analysis
├── pytorchtools.py # PyTorch utility functions
└── README.md
```

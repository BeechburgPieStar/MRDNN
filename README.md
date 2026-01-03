# MRDNN

This repository provides the official implementation of **MRDNN**, a bio inspired multi resolution dynamic neural network for nonlinear behavioral modeling of wideband power amplifiers.

The code corresponds to the paper:

**Power Amplifier Behavioral Modeling Using Bio Inspired Multi Resolution Dynamic Neural Network**

All source code, trained model weights, and training logs are publicly available in this repository.

---

## Overview

Power amplifiers in wideband wireless communication systems exhibit strong nonlinearity and memory effects, leading to spectral regrowth and adjacent channel interference. Although deep learning based behavioral models have achieved promising accuracy, their robustness and adaptability under dynamic signal conditions remain challenging.

MRDNN addresses these issues by integrating:

- A multi resolution interactive fusion module to jointly capture short term nonlinearities and long term memory effects
- A bio inspired dynamic modeling module based on liquid neural networks to adapt to time varying signal environments

The proposed framework focuses not only on time domain modeling accuracy but also on adjacent channel spectral suppression.

---

## Repository Structure

The above code repository implements the complete experimental configuration and result reproduction pipeline corresponding to **MRDNN with GMP** reported in Table V of the paper. The main structure of this repository is as follows:

```
MRDNN/
├── Datasets/ # Dataset from (https://github.com/ITU-AI-ML-in-5G-Challenge/Team-MLAP-solution-for-ML5G-PS-007-Non-linear-Power-Amplifier-Behavioral-Modeling)
├── Model/ # MRDNN model
├── data_save/ # Saved intermediate data and outputs
├── logs/ # Training logs and losses
├── model_save/ # Trained model weights
├── main.py # Training entry script
├── Evaluation.py # Evaluation tools, including NMSE and ACEPR
├── Parameter.py # Hyperparameter configuration
├── data_processing.py # Data preprocessing
├── load_dataset.py # Dataset loading utilities
├── flops.py # FLOPs and complexity analysis
├── pytorchtools.py # PyTorch utility functions
└── README.md
```

## How to run?

```
python main.py
```

## Requirenment

This implementation is based on PyTorch and additionally relies on the [ncps](https://pypi.org/project/ncps/) library for Liquid Neural Network.

## License / 许可证

```
本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
```


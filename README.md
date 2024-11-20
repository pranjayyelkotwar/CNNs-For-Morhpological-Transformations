# Escaping the Matrix : Exploring CNNs for Morphological Transformation Tasks

This repository contains the implementation of the **Improved Morphology Model** designed for classification tasks based on morphological transformations. The project is structured with two primary files:

- `datageneration.py` for generating input-output pairs.
- `main.ipynb` for training and evaluating the model.

The data generated will be stored inside a `datasets/` directory, which will be used directly by the `main.ipynb` notebook.

---

## Project Structure

├── datasets/               # Directory where generated data will be stored  
├── datageneration.py       # Script to generate input-output data pairs  
├── main.ipynb              # Jupyter notebook for training and evaluating the model  
└── README.md               # Project documentation  

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.10 or higher
- Other dependencies: NumPy, Matplotlib, etc.

### Install dependencies

```bash
pip install torch torchvision numpy matplotlib
```
Usage

1. Data Generation

Run datageneration.py to generate input-output pairs:
```bash
python datageneration.py
```

This script will create a datasets/ directory containing the generated dataset.

2. Model Training and Evaluation

Open main.ipynb in Jupyter Notebook and run the cells to:
	•	Load the generated dataset from the datasets/ directory.
	•	Train the ImprovedMorphologyModel.
	•	Evaluate the model’s performance.

Model Overview

The ImprovedMorphologyModel consists of:
	•	Convolutional Layers for feature extraction.
	•	Residual Blocks to enable deeper networks and better learning.
	•	Fully Connected Layers to produce the final predictions.
	•	Dropout and Batch Normalization for regularization and better generalization.

Data Generation (datageneration.py)

The datageneration.py script is responsible for generating the dataset required for training. This script will create a series of input-output image pairs based on predefined morphological transformation sequences, which are stored in the datasets/ directory.

Training (main.ipynb)

In the main.ipynb notebook:
	•	The data generated by datageneration.py is loaded directly from the datasets/ folder.
	•	The model architecture is defined and trained using the data.
	•	The performance of the model is evaluated on validation/test sets.

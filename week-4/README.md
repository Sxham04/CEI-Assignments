# Week 4: CIFAR-10 Image Classification

**Intern:** Soham Sharma  
**LMS ID:** CT_CSI_DS_1031  
**Dataset:** CIFAR-10  


## Objective
Build an image classification model on CIFAR-10 covering the Week 4 topics: perceptron/MLP, activation functions, CNN architecture, training strategies, and evaluation.

## Dataset
CIFAR-10, 60000 32x32 color images across 10 classes (50000 train, 10000 test). Loaded locally from `utils/week4/cifar-10.npz`, not tracked in git.

## Models

| Model | Description |
|---|---|
| ANN baseline | Fully connected network on flattened pixels |
| ANN deep | Deeper ANN with batch norm, dropout, EarlyStopping |
| CNN baseline | Conv layers with batch norm and max pooling |
| CNN deep | More filters, doubled conv blocks per stage, EarlyStopping |
| CNN augmented | CNN deep architecture trained with random flip, rotation, zoom |

## Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| CNN deep | 0.8185 | 0.5439 |
| CNN augmented | 0.7448 | 0.7345 |
| ANN deep | 0.4678 | 1.4921 |

## Key Findings
CNN outperforms ANN by a wide margin since convolution preserves spatial structure in the image. Adding depth and more filters improves the CNN further, but needs EarlyStopping to avoid overfitting over 20 epochs. Data augmentation narrows the gap between train and validation accuracy compared to the plain deep CNN, showing reduced overfitting, though it does not beat the non augmented deep CNN on raw test accuracy in this run.

## How to Run
1. Place `cifar-10.npz` in `utils/week4/`.
2. Run all cells top to bottom. Requires TensorFlow, scikit-learn, pandas, matplotlib.
3. Full run takes about 20 to 25 minutes on GPU.

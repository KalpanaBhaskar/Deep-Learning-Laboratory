
## Experiment 3: Implementation of Convolutional Neural Networks (CNNs) for Image Classification

This repository contains the implementation of a Convolutional Neural Network (CNN) built from scratch using PyTorch to classify images from the CIFAR-10 dataset. The experiment explores core CNN concepts including convolution, padding, stride, pooling, and feature map visualization.

### Objective
To understand the working principles of Convolutional Neural Networks by implementing convolution, pooling, feature map visualization, and image classification using PyTorch.

### Dataset
**CIFAR-10**
- **Training Images:** 50,000
- **Testing Images:** 10,000
- **Image Dimensions:** 32x32x3 (RGB)
- **Classes:** 10 (Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)

### Model Architecture
The network follows a standard feature-extraction to classification pipeline:
`Input(3x32x32) -> Conv2d(16) -> ReLU -> MaxPool2d -> Conv2d(32) -> ReLU -> MaxPool2d -> Flatten -> Linear(10) -> Softmax (via CrossEntropyLoss)`

**Notebook Structure**
The implementation is broken down into 6 logical cells:

- Cell 1: Initialization & Imports: Environment setup, GPU configuration, and global random seeds.

- Cell 2: Dataset Exploration: Downloads CIFAR-10, plots a 2x5 grid of sample images, and visualizes class distribution.

- Cell 3: Convolution & Dimension Analysis: Mathematically calculates and empirically verifies tensor shapes across different kernel sizes, strides, and padding configurations.

- Cell 4: Feature Maps & Pooling: Visualizes 8 feature maps post-convolution and compares the visual effects of Max Pooling vs. Average Pooling.

- Cell 5: CNN Construction & Training: Defines the PyTorch nn.Module and executes the training loop for 20 epochs using the Adam optimizer.

- Cell 6: Evaluation & Visualization: Plots learning curves, calculates test set metrics (Accuracy, Precision, Recall, F1-score), and generates the Confusion Matrix.

### Setup & Requirements
If you are running this locally, install the dependencies using:
```bash
pip install -r requirements.txt

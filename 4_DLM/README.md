# Experiment 4: Comparative Study of Deep CNN Architectures Using Transfer Learning

**Course:** CS3807 – Deep Learning Laboratory  
**Branch:** B.Tech Artificial Intelligence & Data Science | Semester V  
**Dataset:** CIFAR-10  
**AY:** 2026–27

---

## Objective

- Study the evolution of deep CNN architectures (LeNet-5 → ResNet).
- Understand and implement transfer learning and fine-tuning.
- Compare classification performance across architectures.

---

## Dataset – CIFAR-10

| Property | Value |
|---|---|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Classes | 10 (Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck) |
| Image Size | 32 × 32 × 3 |

---

## CNN Architecture Overview

| Model | Year | Depth | Parameters | Key Contribution |
|---|---|---|---|---|
| LeNet-5 | 1998 | 7 | 60K | First practical CNN |
| AlexNet | 2012 | 8 | 61M | ReLU + Dropout + GPU |
| VGG16 | 2014 | 16 | 138M | Deep 3×3 filters |
| GoogleNet | 2014 | 22 | 6.8M | Inception modules |
| ResNet50 | 2015 | 50 | 25.6M | Residual (skip) connections |

---

## Key Concepts

### Convolution Output Dimension

$$O = \left\lfloor \frac{N + 2P - K}{S} \right\rfloor + 1$$

where $N$ = input size, $K$ = kernel size, $P$ = padding, $S$ = stride.

---

### ResNet – Residual Learning

Standard networks learn $H(x)$ directly. ResNet instead learns the **residual**:

$$F(x) = H(x) - x$$

So the final output becomes:

$$\text{Output} = F(x) + x$$

The identity shortcut $x$ lets gradients flow directly, solving the **vanishing gradient problem** in very deep networks.

---

### Dilated (Atrous) Convolution

Increases the receptive field without adding parameters by inserting gaps (zeros) between kernel elements. For dilation rate $D$:

- $D = 1$ → standard convolution  
- $D > 1$ → expanded receptive field

Effective kernel size: $K' = K + (K-1)(D-1)$

---

### Transpose Convolution

Performs **learnable upsampling** — increases spatial dimensions of feature maps. Used in autoencoders, GANs, and semantic segmentation.

---

## Transfer Learning Workflow

```
ImageNet Pretrained Model
        ↓
  Remove Classifier
        ↓
  Freeze Conv Layers
        ↓
   Add Dense Layers
        ↓
  Train Classifier
        ↓
Fine-tune Selected Layers
        ↓
     Prediction
```

**Steps:**
1. Load pretrained model (VGG16 / ResNet50 / MobileNetV2).
2. Remove the original classification head.
3. Freeze convolutional base.
4. Add Global Average Pooling → Dense (ReLU) → Softmax output.
5. Train new classifier layers.
6. Unfreeze last conv block and fine-tune with a smaller learning rate.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 10–20 |
| Loss | Categorical Cross-Entropy |
| Metric | Accuracy |

**Fine-tuning:** Unfreeze last conv block → train 5–10 more epochs at lower LR (e.g., `1e-4` or `1e-5`).

---

## Hyperparameter Study

| Hyperparameter | Values Tested |
|---|---|
| Learning Rate | 0.001, 0.0001 |
| Batch Size | 16, 32, 64 |
| Epochs | 10, 20 |
| Optimizer | Adam, SGD |
| Dense Units | 128, 256 |
| Frozen Layers | All, Partial |

> **Rule:** Change one hyperparameter at a time; keep all others fixed.

---

## Required Plots

| # | Plot |
|---|---|
| 1 | Sample CIFAR-10 images |
| 2 | Training Accuracy vs. Epochs |
| 3 | Validation Accuracy vs. Epochs |
| 4 | Training Loss vs. Epochs |
| 5 | Validation Loss vs. Epochs |
| 6 | Confusion Matrix |
| 7 | Misclassified Images *(optional)* |

> Each plot must include a **2–3 line inference** explaining what is shown, the trend observed, and a reason for it.

---

## Results Summary

### Performance Metrics

| Metric | Value |
|---|---|
| Training Accuracy | |
| Testing Accuracy | |
| Precision | |
| Recall | |
| F1-score | |
| Training Time | |
| Total Parameters | |

### Architecture Comparison

| Model | Parameters | Accuracy (%) | Training Time |
|---|---|---|---|
| LeNet-5 | | | |
| AlexNet | | | |
| VGG16 | | | |
| GoogleNet | | | |
| ResNet50 | | | |

---

## Discussion Questions

1. Why is AlexNet considered a breakthrough in deep learning?
2. Why does VGG16 use only 3×3 convolution filters?
3. Explain the advantages of the Inception module in GoogleNet.
4. What is the purpose of residual learning in ResNet?
5. Differentiate LeNet-5 and ResNet.
6. What is transfer learning and why is it useful?
7. Why is fine-tuning required after feature extraction?
8. Explain the difference between dilated convolution and transpose convolution.
9. Why do pretrained models converge faster?
10. Compare the computational complexity of LeNet and ResNet.

---

## References

1. LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, IEEE, 1998.
2. Krizhevsky et al., *ImageNet Classification with Deep CNNs*, NeurIPS, 2012.
3. Simonyan & Zisserman, *Very Deep Convolutional Networks*, ICLR, 2015.
4. Szegedy et al., *Going Deeper with Convolutions*, CVPR, 2015.
5. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
6. Goodfellow, Bengio & Courville, *Deep Learning*, MIT Press, 2016.
7. TensorFlow Docs: https://www.tensorflow.org
8. Keras Docs: https://keras.io

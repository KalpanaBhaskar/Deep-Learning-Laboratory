## Experiment 4: Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

---

## 1. Objective

The objective of this experiment is to study the evolution of **Deep Convolutional Neural Networks (CNNs)** and experimentally analyze modern CNN architectures using **transfer learning** and **fine-tuning**.

The experiment focuses on understanding the architectural progression from **LeNet-5, AlexNet, VGG16, and GoogleNet (Inception)** to **ResNet**, while evaluating the effectiveness of pretrained deep networks on the **CIFAR-10 image classification dataset**.

The primary objectives are:

* To study the evolution of deep CNN architectures.
* To understand the architectural principles of LeNet-5, AlexNet, VGG16, GoogleNet, and ResNet.
* To understand the mathematical foundations of convolution, pooling, activation, and residual learning.
* To implement transfer learning using an ImageNet-pretrained CNN.
* To freeze and subsequently fine-tune selected layers of a pretrained network.
* To evaluate classification performance using accuracy, precision, recall, F1-score, and a confusion matrix.
* To analyze the effect of hyperparameters such as learning rate, batch size, optimizer, and number of trainable layers.
* To compare the computational characteristics and classification performance of different CNN architectures.

---

## 2. Theoretical Background

### 2.1 Convolutional Neural Networks

A Convolutional Neural Network is a hierarchical neural architecture designed primarily for processing grid-structured data such as images.

For an input feature map $X$ and convolution kernel $K$, a two-dimensional convolution can be expressed as:

$$Y(i,j)=\sum_{m}\sum_{n}X(i+m,j+n)K(m,n)$$

where:

* $X$ is the input feature map,
* $K$ is the learnable convolution kernel,
* $Y$ is the resulting feature map.

In a CNN, multiple convolutional filters learn different spatial patterns such as edges, textures, shapes, and higher-level semantic features. A typical CNN can be represented as:

$$\text{Input} \rightarrow \text{Convolution} \rightarrow \text{Activation} \rightarrow \text{Pooling} \rightarrow \text{Deep Feature Extraction} \rightarrow \text{Classifier}$$

The hierarchical representation learned by CNNs can be viewed as:

$$\text{Pixels} \rightarrow \text{Edges} \rightarrow \text{Textures} \rightarrow \text{Shapes} \rightarrow \text{Objects}$$

---

### 2.2 Activation Function

Modern CNN architectures commonly use the **Rectified Linear Unit (ReLU)** activation:

$$\operatorname{ReLU}(x)=\max(0,x)$$

ReLU introduces non-linearity while reducing the vanishing-gradient problem associated with saturating activation functions. For the output layer of a multi-class classifier, the **Softmax** function is used:

$$P(y=i\vert{}x)=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}$$

where $C$ is the number of classes and $z_i$ is the logit corresponding to class $i$. For CIFAR-10, $C=10$, and therefore the output layer produces a probability vector:

$$\mathbf{p}=[p_1,p_2,\ldots,p_{10}]$$

such that:

$$\sum_{i=1}^{10}p_i=1$$

---

## 3. Evolution of CNN Architectures

The development of CNN architectures has progressively focused on increasing representational capacity while improving optimization efficiency and computational performance.

| Architecture | Year | Approx. Depth | Major Contribution |
| --- | --- | --- | --- |
| **LeNet-5** | 1998 | 7 | Early practical CNN architecture |
| **AlexNet** | 2012 | 8 | ReLU, Dropout, GPU training |
| **VGG16** | 2014 | 16 | Deep architecture with $3 \times 3$ convolutions |
| **GoogleNet** | 2014 | 22 | Inception modules and multi-scale features |
| **ResNet-50** | 2015 | 50 | Residual/skip connections |

The progression can be summarized as:

$$\text{LeNet} \rightarrow \text{AlexNet} \rightarrow \text{VGG} \rightarrow \text{GoogleNet} \rightarrow \text{ResNet}$$

where each generation addresses limitations of previous architectures.

---

## 4. Major CNN Architectures

### 4.1 LeNet-5

LeNet-5 was introduced by Yann LeCun and collaborators for handwritten digit recognition. Its architecture consists primarily of convolutional layers, subsampling/pooling layers, and fully connected layers. Its significance lies in demonstrating that CNNs could automatically learn useful visual representations directly from image data. The relatively small number of parameters makes LeNet computationally inexpensive; however, its shallow architecture limits its ability to learn highly complex visual representations.

---

### 4.2 AlexNet

AlexNet achieved a major improvement in large-scale image classification performance in the 2012 ImageNet competition. Its important innovations include:

* ReLU activation,
* Dropout regularization,
* GPU-based training,
* Data augmentation,
* Deep convolutional feature extraction.

The introduction of ReLU can be expressed as $f(x)=\max(0,x)$, which enables efficient gradient-based optimization. Dropout further reduces overfitting by randomly deactivating a subset of neurons during training.

---

### 4.3 VGG16

VGG16 increased network depth while maintaining a simple and uniform architecture based primarily on $3 \times 3$ convolutional filters. Two consecutive $3 \times 3$ convolutions provide an effective receptive field comparable to a larger filter while introducing additional nonlinear transformations.

For example, $3 \times 3 + 3 \times 3$ provides an effective receptive field of $5 \times 5$. The use of small kernels therefore allows VGG-style networks to construct deep hierarchical representations while maintaining architectural simplicity. The major limitation of VGG16 is its large number of parameters and consequently high computational and memory requirements.

---

### 4.4 GoogleNet / Inception

GoogleNet introduced the **Inception module**, which performs feature extraction at multiple spatial scales within the same layer. An idealized Inception module can be represented as:

$$\operatorname{Inception}(x) = \operatorname{Concat}\left[f_{1\times1}(x), f_{3\times3}(x), f_{5\times5}(x), f_{\text{pool}}(x)\right]$$

The resulting feature maps are concatenated along the channel dimension. The $1 \times 1$ convolution is particularly important because it performs channel-wise dimensionality reduction before computationally expensive convolutions. Thus, Inception improves the trade-off between Representation Capacity and Computational Cost.

---

### 4.5 ResNet

As CNNs become deeper, optimization becomes increasingly difficult due to degradation and gradient-flow problems. ResNet addresses this limitation using **residual learning**. Instead of directly learning $H(x)$, a residual block learns:

$$F(x)=H(x)-x$$

The final output is therefore:

$$H(x)=F(x)+x$$

The identity mapping $x$ is transmitted through a **skip connection**, allowing gradients to propagate more effectively through deep networks. A residual block can therefore be represented as:

$$x \rightarrow \boxed{\text{Conv} \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{Conv} \rightarrow \text{BN}} \rightarrow + x \rightarrow \text{ReLU}$$

This formulation enables the construction of substantially deeper networks such as ResNet-50, ResNet-101, and ResNet-152.

---

## 5. Transfer Learning

Training a deep CNN from randomly initialized parameters requires a large amount of labeled data and computational resources. **Transfer learning** addresses this problem by reusing knowledge learned from a large source dataset. In this experiment, an ImageNet-pretrained model is used as the source network.

Let $f_{\theta}(x)$ denote a pretrained CNN with parameters $\theta$. The network can be decomposed into:

$$f_{\theta}(x) = g_{\phi}(h_{\psi}(x))$$

where:

* $h_{\psi}$ represents the convolutional feature extractor,
* $g_{\phi}$ represents the classification head.

During transfer learning, the original classifier is replaced with a new classifier suitable for CIFAR-10. The resulting architecture is:

$$\boxed{\text{Input} \rightarrow \text{Pretrained CNN} \rightarrow \text{Global Average Pooling} \rightarrow \text{Dense} \rightarrow \text{Softmax}}$$

The pretrained feature extractor initially remains frozen:

$$\psi=\text{constant}$$

Only the newly initialized classifier parameters $\phi$ are optimized.

---

## 6. Global Average Pooling

Instead of flattening the entire convolutional feature map, **Global Average Pooling (GAP)** computes the spatial average of each feature channel. For feature map $k$:

$$z_k = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} A_{ijk}$$

Thus, an $H \times W \times K$ feature tensor is transformed into a $K$-dimensional feature vector. GAP significantly reduces the number of trainable parameters compared with large fully connected layers and can reduce overfitting.

---

## 7. Fine-Tuning

After training the newly added classifier, selected layers of the pretrained convolutional base are unfrozen. The process can be expressed as:

$$\theta = [\theta_{\text{frozen}},\theta_{\text{trainable}}]$$

During initial transfer learning:

$$\nabla_{\theta_{\text{frozen}}}L=0$$

During fine-tuning, selected deeper layers become trainable:

$$\nabla_{\theta_{\text{trainable}}}L\neq0$$

The learning rate is generally reduced during fine-tuning so that previously learned representations are not destroyed by large parameter updates. The experimental comparison is therefore:

$$\boxed{\text{Frozen Feature Extractor} \longrightarrow \text{Fine-Tuned Feature Extractor}}$$

and the corresponding change in test accuracy is analyzed.

---

## 8. CIFAR-10 Dataset

The experiment uses the **CIFAR-10** dataset, which contains 50,000 training images and 10,000 test images distributed across 10 classes. Each image has dimensions $32 \times 32 \times 3$, where the three channels correspond to RGB color components.

The ten classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The pixel values are normalized from `[0, 255]` to `[0, 1]` using:

$$x_{\text{normalized}}=\frac{x}{255}$$

Normalization improves numerical stability during optimization.

---

## 9. Optimization Objective

For a $C$-class classification problem, the categorical cross-entropy loss is:

$$L = -\sum_{i=1}^{C}y_i\log(\hat{y}_i)$$

where:

* $y_i$ is the true one-hot encoded label,
* $\hat{y}_i$ is the predicted probability,
* $C=10$ for CIFAR-10.

The model parameters are optimized using the **Adam optimizer** with an initial learning rate of $\eta=0.001$. The parameter update can be conceptually represented as:

$$\theta_{t+1} = \theta_t-\eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

The experimental batch size is $B=32$, with an initial training duration of approximately 10 to 20 epochs.

---

## 10. Performance Evaluation

The trained model is evaluated using multiple classification metrics.

### Accuracy

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$


For multi-class classification, accuracy represents the fraction of correctly classified samples.

### Precision

$$\text{Precision} = \frac{TP}{TP+FP}$$


Precision measures the reliability of positive predictions.

### Recall

$$\text{Recall} = \frac{TP}{TP+FN}$$


Recall measures the ability of the classifier to correctly identify samples belonging to a class.

### F1-score

$$F_1 = 2\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}$$


The F1-score provides a harmonic mean of precision and recall.

### Confusion Matrix

For $C=10$ classes, the confusion matrix is $M\in\mathbb{R}^{10\times10}$, where $M_{ij}$ represents the number of samples belonging to class $i$ that were predicted as class $j$. The diagonal elements represent correct predictions, while off-diagonal elements represent classification errors.

---

## 11. Experimental Workflow

The complete experimental pipeline is organized as follows:

$$\boxed{\text{CIFAR-10} \rightarrow \text{Preprocessing} \rightarrow \text{Pretrained CNN} \rightarrow \text{Classifier Replacement} \rightarrow \text{Frozen Training} \rightarrow \text{Fine-Tuning} \rightarrow \text{Evaluation}}$$

The workflow consists of:

1. Load the CIFAR-10 dataset.
2. Normalize image pixel values.
3. Visualize representative samples.
4. Resize images to the input dimensions required by the selected pretrained model.
5. Load ImageNet-pretrained CNN weights.
6. Remove the original classification head.
7. Freeze the convolutional base.
8. Add Global Average Pooling.
9. Add a fully connected layer with ReLU activation.
10. Add a 10-class Softmax output layer.
11. Train the newly added classifier.
12. Unfreeze the final convolutional block.
13. Fine-tune the selected layers.
14. Evaluate the model on the test dataset.
15. Generate accuracy and loss curves.
16. Generate the confusion matrix and classification report.
17. Analyze misclassified samples.
18. Compare performance before and after fine-tuning.
19. Study the effect of selected hyperparameters.

---

## 12. Hyperparameter Study

The following hyperparameters are investigated experimentally:

| Hyperparameter | Values |
| --- | --- |
| Learning Rate | 0.001, 0.0001 |
| Batch Size | 16, 32, 64 |
| Epochs | 10, 20 |
| Optimizer | Adam, SGD |
| Dense Units | 128, 256 |
| Frozen Layers | All, Partial |

The objective of the hyperparameter study is to analyze how optimization settings and the degree of fine-tuning influence convergence and generalization.

---

## 13. Experimental Comparison

The architectures considered in the theoretical comparison are:

| Model | Depth | Approx. Parameters | Key Innovation |
| --- | --- | --- | --- |
| LeNet-5 | 7 | ~60K | Early CNN architecture |
| AlexNet | 8 | ~61M | ReLU + Dropout |
| VGG16 | 16 | ~138M | $3 \times 3$ convolutions |
| GoogleNet | 22 | ~6.8M | Inception modules |
| ResNet-50 | 50 | ~25.6M | Residual learning |

The comparison demonstrates that **greater depth does not necessarily imply a proportionally greater parameter count**. For example, GoogleNet achieves considerable depth while using substantially fewer parameters than VGG16 due to its computationally efficient Inception design.

---

## 14. Expected Analysis

The experiment is designed to investigate the following relationships:

$$\text{Architecture} \rightarrow \text{Feature Representation} \rightarrow \text{Optimization} \rightarrow \text{Generalization} \rightarrow \text{Classification Performance}$$

Particular attention is given to:

* convergence behavior,
* training versus validation accuracy,
* training versus validation loss,
* effect of freezing pretrained layers,
* effect of fine-tuning,
* computational complexity,
* class-wise performance,
* confusion between visually similar classes.

The central hypothesis is that pretrained CNNs provide a strong initialization because early and intermediate layers contain transferable visual representations. Fine-tuning selected deeper layers allows these representations to adapt to the target dataset while requiring substantially less optimization than training the entire network from scratch.

---

## 15. Notebook Structure

This notebook follows the experimental sequence:

**Dataset Preparation → Visualization → Preprocessing → Transfer Learning → Model Training → Fine-Tuning → Evaluation → Hyperparameter Analysis → Comparative Study → Discussion**

The subsequent sections contain the implementation, training results, mandatory plots, performance metrics, confusion matrix, and experimental observations.

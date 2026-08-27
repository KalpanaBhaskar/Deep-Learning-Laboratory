# MobileNetV2 Optimization and Transfer Learning on Oxford-IIIT Pet Dataset

**Course:** CS3807 – Deep Learning Laboratory | **Branch:** B.Tech AI & DS (Semester V)

This repository contains the implementation and systematic analysis of deep learning techniques—ranging from weight initialization and regularization to hyperparameter tuning and transfer learning. We utilize **MobileNetV2** to perform image classification on the **Oxford-IIIT Pet Dataset**.

---

## 📌 Project Objective

To systematically study the impact of various deep learning training strategies on model performance. The experiments isolate and evaluate weight initialization, regularization, optimizers, learning rate scheduling, transfer learning (feature extraction vs. fine-tuning), and K-Fold cross-validation.

---

## 🗄️ Dataset Details

| Property | Value |
| --- | --- |
| **Dataset** | Oxford-IIIT Pet Dataset |
| **Classes** | 37 distinct cat and dog breeds |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Normalization** | Scaled as per MobileNetV2 pretrained requirements |

> **Note:** A dedicated test set was held out and remained completely untouched during all hyperparameter selection and cross-validation stages to ensure unbiased final evaluation.

---

## 🧠 Model Architecture: MobileNetV2

The project uses MobileNetV2, chosen for its computational efficiency. Key architectural features include depthwise separable convolutions, inverted residuals, linear bottlenecks, Batch Normalization, and ReLU6 activations.

**Simplified Forward Pass:**

```text
Input (224×224×3) → Conv → Depthwise Conv → Pointwise Conv (1×1) 
→ Inverted Residual Blocks → Global Average Pooling → Dense → Softmax

```

---

## 🔬 Experiments Conducted

The codebase is modularized to run the following experimental stages, generating comparative plots and logs for each.

### 1. Weight Initialization

We explore how weight initialization impacts symmetry breaking and convergence. The repository contains training/validation curves comparing:

* **Zero:** All weights = 0 (baseline to demonstrate failure).
* **Random:** Small random values.
* **Xavier / Glorot:** Suited for sigmoid/tanh activations.

$$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$


* **He:** Suited for ReLU activations.

$$\text{Var}(W) = \frac{2}{n_{in}}$$



### 2. Regularization & Overfitting

To bridge the generalization gap (difference between train and test performance), we evaluate:

* **Baseline:** No regularization.
* **L2 (Weight Decay):** Penalizes large weights by adding a penalty to the loss:

$$\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_{l} \Vert{}W^{(l)}\Vert{}^2$$


* **Dropout:** Randomly zeros activations with probability $p$.

### 3. Batch Normalization (BN)

We evaluate the stabilizing effect of Batch Normalization on training dynamics. For a mini-batch $\{x_1, x_2, \ldots, x_m\}$, BN computes:

* **Mean:** $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$
* **Variance:** $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$
* **Normalized activation:** $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
* **Scaled & shifted output:** $y_i = \gamma \hat{x}_i + \beta$ (where $\gamma$ and $\beta$ are learnable).

### 4. Optimization Algorithms

The repository tracks convergence time, final loss, and validation accuracy across four optimizers:

* **SGD:** $W \leftarrow W - \eta \nabla \mathcal{L}$
* **Momentum:** $v \leftarrow \beta v + \nabla \mathcal{L};\quad W \leftarrow W - \eta v$
* **RMSProp:** Adapts learning rate based on a moving average of squared gradients.
* **Adam:** Combines Momentum and RMSProp with bias correction:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla\mathcal{L}, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla\mathcal{L})^2$$


$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$


$$W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t$$



### 5. Hyperparameter Tuning

Hyperparameters were tuned iteratively (changing one at a time). Convolutional output dimensions were calculated via:


$$O = \left\lfloor \frac{N + 2P - K}{S} \right\rfloor + 1$$


*(where $N$=input, $K$=kernel, $P$=padding, $S$=stride)*

**Search Space:**

* **Learning Rate:** 0.001, 0.0001
* **Batch Size:** 16, 32, 64
* **Dropout Rate:** 0, 0.25, 0.5

### 6. Transfer Learning & Fine-Tuning

Two transfer learning paradigms were implemented:

* **Case A (Feature Extraction):** Pretrained base frozen; only the new dense classifier is trained.
* **Case B (Fine-Tuning):** Top layers of the base unfrozen and trained with a very small learning rate ($\eta_{FT} \ll \eta_{base}$) to avoid catastrophic forgetting of learned features.

### 7. K-Fold Cross-Validation (K=5)

To ensure robustness, the best configuration was validated using 5-Fold CV. Results in the logs are reported as Mean ± Standard Deviation:

* **Mean:** $\bar{A} = \frac{1}{5} \sum_{i=1}^{5} A_i$
* **SD:** $SD = \sqrt{\frac{1}{5} \sum_{i=1}^{5} (A_i - \bar{A})^2}$

---

## 📊 Results & Analysis

All generated plots, tables, and analytical inferences can be found in the `results/` directory and the accompanying Jupyter Notebooks.

**What's included in the outputs:**

* **Plots:** Training/Validation loss and accuracy curves for every experiment (Optimizers, Regularization, Init strategies).
* **Final Evaluation:** The chosen model trained on the full train set and evaluated on the untouched test set.
* **Metrics Logs:** Precision, Recall, F1-Score, Parameter counts, and Confusion Matrices.
* **Inferences & Discussion:** A detailed `report.pdf` (or markdown file) is included in this repository. It provides 2-3 line inferences for every generated plot, explaining the observed trends and answering theoretical questions regarding model behavior, parameter dynamics, and algorithmic choices.

4. Parkhi et al., *Cats and Dogs*, CVPR, 2012.
5. TensorFlow Documentation: [https://www.tensorflow.org](https://www.tensorflow.org)
6. Keras Documentation: [https://keras.io](https://keras.io)

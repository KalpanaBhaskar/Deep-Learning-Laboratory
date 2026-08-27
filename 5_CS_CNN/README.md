# Experiment 5: CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning & Cross-Validation

**Course:** CS3807 – Deep Learning Laboratory  
**Branch:** B.Tech Artificial Intelligence & Data Science | Semester V  
**Model:** MobileNetV2 | **Dataset:** Oxford-IIIT Pet Dataset  
**AY:** 2026–27

---

## Objective

Systematically study the effect of weight initialization, regularization, optimization, hyperparameter tuning, transfer learning, and cross-validation on image classification using **MobileNetV2** on the **Oxford-IIIT Pet Dataset**.

---

## Dataset – Oxford-IIIT Pet Dataset

| Property | Value |
|---|---|
| Classes | 37 cat/dog breeds |
| Input Size | 224 × 224 × 3 (RGB) |
| Normalization | Per pretrained model requirements |

> The **test set must remain untouched** throughout all hyperparameter selection stages.

---

## MobileNetV2 Architecture (Simplified)

```
Input (224×224×3)
    → Conv
    → Depthwise Conv
    → Pointwise Conv (1×1)
    → Inverted Residual Blocks (with Linear Bottlenecks)
    → Global Average Pooling
    → Dense
    → Softmax
```

Key features: depthwise separable convolutions, inverted residuals, linear bottlenecks, Batch Normalization, ReLU6.

---

## 1. Weight Initialization

| Strategy | Description |
|---|---|
| Zero | All weights = 0 (problematic — symmetry breaking fails) |
| Random | Small random values |
| Xavier / Glorot | $W \sim \mathcal{U}\!\left[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}},\ \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\right]$ — suited for sigmoid/tanh |
| He | $W \sim \mathcal{N}\!\left(0,\ \frac{2}{n_{in}}\right)$ — suited for ReLU |

**Xavier initialization variance:**

$$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$

**He initialization variance:**

$$\text{Var}(W) = \frac{2}{n_{in}}$$

**Plots required:**  
- Plot 1: Training Loss vs. Epoch (one curve per method)  
- Plot 2: Validation Accuracy vs. Epoch (one curve per method)

---

## 2. Regularization & Overfitting

Overfitting: model performs well on training data but poorly on unseen data (large **generalization gap**).

| Technique | Effect |
|---|---|
| No regularization | Baseline; prone to overfitting |
| L2 (Weight Decay) | Adds $\lambda \|W\|^2$ penalty to loss; penalizes large weights |
| Dropout | Randomly zeros activations with probability $p$ during training |
| Batch Normalization | Normalizes activations; acts as implicit regularizer |

**L2 regularized loss:**

$$\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_{l} \|W^{(l)}\|^2$$

**Plots required:**  
- Plot 3: Train & Validation Accuracy vs. Epoch  
- Plot 4: Train & Validation Loss vs. Epoch

---

## 3. Batch Normalization

For a mini-batch $\{x_1, x_2, \ldots, x_m\}$:

**Batch mean:**

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**Batch variance:**

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

**Normalized activation:**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Scaled & shifted output:**

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma$ (scale) and $\beta$ (shift) are **learnable parameters**.

**Numerical Example** for $x = [2, 4, 6, 8]$:

$$\mu_B = 5, \quad \sigma_B^2 = 5, \quad \sqrt{\sigma_B^2} \approx 2.236$$

$$\hat{x} \approx [-1.342,\ -0.447,\ 0.447,\ 1.342]$$

With $\gamma = 1,\ \beta = 0$, output $\approx [-1.342,\ -0.447,\ 0.447,\ 1.342]$.

**Plot required:**  
- Plot 5: Validation Accuracy — With BN vs. Without BN

---

## 4. Optimization Algorithms

| Optimizer | Update Rule |
|---|---|
| SGD | $W \leftarrow W - \eta \nabla \mathcal{L}$ |
| Momentum | $v \leftarrow \beta v + \nabla \mathcal{L};\quad W \leftarrow W - \eta v$ |
| RMSProp | $W \leftarrow W - \frac{\eta}{\sqrt{E[g^2] + \epsilon}}\nabla \mathcal{L}$ |
| Adam | Combines Momentum + RMSProp with bias correction |

**Adam update:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla\mathcal{L}, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla\mathcal{L})^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t$$

**Plots required:**  
- Plot 6: Training Loss vs. Epoch (per optimizer)  
- Plot 7: Validation Accuracy vs. Epoch (per optimizer)

| Optimizer | Final Loss | Best Val. Acc | Epoch to Converge | Time |
|---|---|---|---|---|
| SGD | | | | |
| Momentum | | | | |
| RMSProp | | | | |
| Adam | | | | |

---

## 5. CNN Hyperparameter Tuning

**Convolution output size:**

$$O = \left\lfloor \frac{N + 2P - K}{S} \right\rfloor + 1$$

where $N$ = input size, $K$ = kernel size, $P$ = padding, $S$ = stride.

| Hyperparameter | Values Tested |
|---|---|
| Learning Rate | 0.001, 0.0001 |
| Batch Size | 16, 32, 64 |
| Dropout Rate | 0, 0.25, 0.5 |
| Optimizer | SGD, Adam |
| Fine-Tuning LR | $10^{-4}$, $10^{-5}$ |
| Frozen Layers | Frozen base / Partial unfreezing |

> **Rule:** Change one hyperparameter at a time; keep all others fixed.

**Plots required:**  
- Plot 8: Learning Rate vs. Validation Accuracy  
- Plot 9: Batch Size vs. Validation Accuracy  
- Plot 10: Dropout Rate vs. Validation Accuracy

---

## 6. Transfer Learning & Fine-Tuning

### Case A – Feature Extraction
```
Pretrained MobileNetV2 → Freeze Base → New Classifier (train only)
```

### Case B – Fine-Tuning
```
Pretrained MobileNetV2 → Unfreeze Top Layers → Train with small LR
```

Fine-tuning uses a smaller learning rate ($\eta_{FT} \ll \eta_{base}$) to avoid destroying pretrained features.

**Plots required:**  
- Plot 11: Validation Accuracy — Feature Extraction vs. Fine-Tuning  
- Plot 12: Training & Validation Loss — before and after fine-tuning

---

## 7. K-Fold Cross-Validation (K = 5)

**Mean accuracy:**

$$\bar{A} = \frac{1}{5} \sum_{i=1}^{5} A_i$$

**Standard deviation:**

$$SD = \sqrt{\frac{1}{5} \sum_{i=1}^{5} (A_i - \bar{A})^2}$$

Report result as $\bar{A} \pm SD$.

| Configuration | F1 | F2 | F3 | F4 | F5 | Mean ± SD |
|---|---|---|---|---|---|---|
| C1 | | | | | | |
| C2 | | | | | | |
| C3 | | | | | | |
| C4 | | | | | | |

**Plot required:**  
- Plot 13: Mean Validation Accuracy per configuration with error bars (±SD)

---

## 8. Final Model Evaluation

After selecting the best configuration via cross-validation:
1. Retrain on the **complete** training set.
2. Evaluate on the **held-out test set** (used only once).

| Metric | Value |
|---|---|
| Mean CV Accuracy | |
| CV Standard Deviation | |
| Test Accuracy | |
| Precision | |
| Recall | |
| F1-score | |
| Training Time | |
| Number of Parameters | |

**Plot required:**  
- Plot 14: Confusion Matrix  
- Plot 15 *(optional)*: Misclassified images with brief explanation

---

## Overall Results Summary

| Configuration | CV Accuracy | SD | Test Accuracy | Training Time |
|---|---|---|---|---|
| Baseline | | | | |
| Best Initialization | | | | |
| Best Regularization | | | | |
| Best Optimizer | | | | |
| Best Hyperparameters | | | | |
| Fine-Tuned Model | | | | |

---

## Inference Requirement

For **every** plot, write a short inference (~2–3 lines) covering:
1. What does the plot show?
2. What trend is observed?
3. Why might the trend occur?

---

## Discussion Questions

1. What is the difference between model parameters and hyperparameters?
2. Why is weight initialization important?
3. Why can zero initialization be problematic?
4. Compare Xavier and He initialization.
5. How do training/validation curves reveal overfitting?
6. How does Dropout reduce overfitting?
7. What is the purpose of Batch Normalization?
8. Explain the Batch Normalization numerical example.
9. What are the roles of $\gamma$ and $\beta$ in Batch Normalization?
10. Compare SGD, Momentum, RMSProp and Adam.
11. What happens when the learning rate is too large / too small?
12. What is the effect of increasing batch size?
13. Explain stride and padding.
14. Why is MobileNetV2 computationally efficient?
15. What is depthwise separable convolution?
16. What is transfer learning?
17. Differentiate feature extraction and fine-tuning.
18. Why is a smaller learning rate used during fine-tuning?
19. Why is K-Fold Cross-Validation useful for hyperparameter selection?
20. Why must the test set remain untouched during tuning?
21. Why should both mean and standard deviation be reported?
22. Is the highest validation accuracy always sufficient to select a model?

---

## References

1. Goodfellow, Bengio & Courville, *Deep Learning*, MIT Press, 2016.
2. Ioffe & Szegedy, *Batch Normalization*, ICML, 2015.
3. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR, 2018.
4. Parkhi et al., *Cats and Dogs*, CVPR, 2012.
5. TensorFlow Docs: https://www.tensorflow.org
6. Keras Docs: https://keras.io

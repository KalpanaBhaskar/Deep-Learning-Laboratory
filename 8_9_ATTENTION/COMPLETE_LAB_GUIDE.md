# CS3807 Deep Learning Laboratory - Complete Implementation Guide

## Overview
This is a comprehensive implementation guide for Experiments 8 & 9: "Effect of Attention Type and Attention Position in a Pretrained ResNet50"

---

## 📁 FILES CREATED

### 1. **LAB_CHECKLIST.md** ✓
   **Comprehensive 500+ item checklist organized into 16 major sections:**
   
   **Contents:**
   - Part 1: Core Experiment Requirements (6 main experiments + mandatory minimum)
   - Part 2: Dataset & Preprocessing
   - Part 3: Architecture Requirements (ResNet50 analysis + attention implementations)
   - Part 4: Training Configuration
   - Part 5: Evaluation Metrics (accuracy, precision, recall, F1, parameters, inference time)
   - Part 6: Main Result Tables (7 different result table formats)
   - Part 7: Graphical Analysis (12 required/recommended plots)
   - Part 8: Confusion Matrix Requirements
   - Part 9: Qualitative Analysis (internal attention maps + Grad-CAM)
   - Part 10: Required Visualizations Checklist (14 visualizations)
   - Part 11: Calculations & Formulas (complete mathematical formulations)
   - Part 12: Discussion Questions (35 critical thinking questions)
   - Part 13: Additional Exploration Tasks
   - Part 14: Required Report Sections
   - Part 15: Experimental Protocol Checklist
   - Part 16: Key Insights & Expected Findings

### 2. **CS3807_Attention_Lab_Part1.ipynb** ✓
   **Setup, Dataset Loading, and Preprocessing (16 cells)**
   
   **Sections:**
   - Cell 1-4: Environment setup & imports
   - Cell 5-7: TensorFlow Flowers dataset loading & exploration
   - Cell 8-10: Image preprocessing & resizing to 224×224×3
   - Cell 11-13: Data splitting (70-15-15) with stratification
   - Cell 12: Data augmentation pipeline (5 techniques)
   - Cell 13: Save splits for reproducibility
   - Cell 14-16: ResNet50 architecture analysis
   
   **Key Features:**
   - Proper seed setting (np.random.seed(42), tf.random.set_seed(42))
   - ImageNet preprocessing for ResNet50
   - Stratified train/val/test split
   - **CRITICAL**: Same augmentation for ALL models
   - ResNet50 feature hierarchy documentation
   - Attention insertion point mapping

### 3. **CS3807_Attention_Lab_Part2.ipynb** ✓
   **Attention Mechanism Implementations (14 cells)**
   
   **Implemented Mechanisms:**
   - Cell 1: Library imports & data loading
   - Cell 2-3: **Channel Attention**
     - Formula: M_c(F) = σ(W_2 δ(W_1 z))
     - Global average pooling for channel descriptor
     - FC layers with ReLU and Sigmoid
   
   - Cell 4-5: **Spatial Attention**
     - Formula: M_s(F) = σ(Conv_7×7([F_avg ∥ F_max]))
     - Average and max pooling across channels
     - 7×7 convolution with sigmoid
   
   - Cell 6-7: **Sequential Channel-Spatial Attention**
     - Flow: F → Channel Attention → Spatial Attention → F_cs
     - Apply channel first, then spatial
   
   - Cell 8-9: **Self-Attention**
     - Formula: A = softmax(QK^T / √d), Y = AV
     - Residual connection: F_SA = F + γ reshape(Y)
     - Dimension scaling to reduce computation
   
   - Cell 10: **Reverse Attention** (Spatial→Channel)
     - For Experiment 4: Attention Order Ablation
   
   - Cell 11: **Variable Spatial Attention**
     - Configurable kernel sizes (3×3, 5×5, 7×7)
     - For Experiment 5: Kernel Size Ablation
   
   - Cell 12: **Combined Attention**
     - Channel + Spatial + Self sequentially
   
   - Cell 13: **SE Attention** (Extended experiment)
     - Squeeze-and-Excitation reference implementation
   
   - Cell 14: Module registry for reuse

### 4. **CS3807_Attention_Lab_Part3.ipynb** ✓
   **Model Construction and Experiments 1-2 (11 cells)**
   
   **Contents:**
   - Cell 1: Load prerequisites
   - Cell 2: Attention module definitions (simplified)
   - Cell 3: Model construction with attention insertion
   - Cell 4: Test model building
   
   **Experiment 1: Attention-Type Ablation (Fixed at P3)**
   - Cell 5: Training configuration (identical for all models)
   - Cell 6: Define 6 models:
     - A0: Baseline ResNet50
     - A1: Channel Attention
     - A2: Spatial Attention
     - A3: Channel+Spatial
     - A4: Self-Attention
     - A5: Combined (Channel+Spatial+Self)
   - Cell 7: Training function with early stopping & LR scheduling
   - Cell 8: Train all Experiment 1 models
   
   **Experiment 2: Attention-Position Ablation (Fixed: Channel→Spatial)**
   - Cell 9: Define 3 models at different positions:
     - B2: Channel+Spatial at P2
     - B3: Channel+Spatial at P3
     - B4: Channel+Spatial at P4
   - Cell 10: Train all Experiment 2 models
   - Cell 11: Save histories and models

**TRAINING CONFIGURATION (IDENTICAL FOR ALL MODELS):**
```
- Batch size: 32
- Epochs: 30 (with early stopping)
- Initial learning rate: 10^-3
- Fine-tuning learning rate: 10^-4
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau
```

---

## 🔄 WORKFLOW TO COMPLETE THE LAB

### Phase 1: Data Preparation (Part 1)
```
1. Run Part1.ipynb Cell 1-16
   ✓ Downloads TensorFlow Flowers
   ✓ Resizes to 224×224×3
   ✓ Applies ImageNet preprocessing
   ✓ Splits 70-15-15 with stratification
   ✓ Applies augmentation pipeline
   ✓ Analyzes ResNet50 architecture
```

### Phase 2: Implement Attention (Part 2)
```
2. Run Part2.ipynb Cell 1-14
   ✓ Defines ChannelAttention
   ✓ Defines SpatialAttention
   ✓ Defines Self-Attention
   ✓ Defines Sequential combinations
   ✓ Tests all modules
   ✓ Creates SE, ECA, CBAM references
```

### Phase 3: Train Baseline & Main Experiments (Part 3)
```
3. Run Part3.ipynb Cell 1-11
   ✓ Loads data and attention modules
   ✓ Builds model construction function
   ✓ Trains Experiment 1 (6 models, type ablation)
   ✓ Trains Experiment 2 (3 models, position ablation)
   ✓ Saves all models and histories
```

### Phase 4: Evaluation & Metrics (Part 4 - TO BE CREATED)
```
4. Create Part4.ipynb:
   ✓ Cell 1-5: Load trained models and test data
   ✓ Cell 6-15: Calculate evaluation metrics
     - Accuracy, Precision, Recall, F1
     - Confusion matrices
     - Per-class analysis
   ✓ Cell 16-20: Generate result tables
     - Table 1: Main result table (9 models)
     - Table 2: Attention-type ablation
     - Table 3: Position ablation
     - Table 4-7: Extended experiments
```

### Phase 5: Visualizations (Part 5 - TO BE CREATED)
```
5. Create Part5.ipynb:
   ✓ Cell 1-10: Training curves
     - Plot 1: Accuracy vs Epoch (train/val)
     - Plot 2: Loss vs Epoch (train/val)
   ✓ Cell 11-15: Comparative plots
     - Plot 3: Mechanism vs Accuracy (bar graph)
     - Plot 4: Mechanism vs F1 (bar graph)
     - Plot 5: Parameters vs Accuracy (scatter)
     - Plot 6: Inference Time vs F1 (scatter)
   ✓ Cell 16-25: Attention-specific plots
     - Per-class accuracy
     - Parameter count comparison
     - Inference time comparison
```

### Phase 6: Grad-CAM & Attention Maps (Part 6 - TO BE CREATED)
```
6. Create Part6.ipynb:
   ✓ Cell 1-10: Implement Grad-CAM
   ✓ Cell 11-20: Channel attention visualization
   ✓ Cell 21-30: Spatial attention heatmaps
   ✓ Cell 31-40: Self-attention responses
   ✓ Cell 41-50: Attention maps at P2, P3, P4
   ✓ Cell 51-60: Comparative visualizations
```

### Phase 7: Additional Experiments (OPTIONAL)
```
7. Implement additional experiments:
   - Experiment 3: Multiple attention positions (P3, P4, P3+P4)
   - Experiment 4: Attention order (Channel→Spatial vs Spatial→Channel)
   - Experiment 5: Kernel size (3×3, 5×5, 7×7)
   - Extended: SE, ECA, CBAM comparisons
```

---

## 📊 EXPERIMENTS CHECKLIST

### MANDATORY EXPERIMENTS (Minimum Set - Section 35)
- [x] **Experiment 1**: Attention-Type Ablation (6 models)
  - [x] A0: Baseline
  - [x] A1: Channel Attention
  - [x] A2: Spatial Attention
  - [x] A3: Channel+Spatial
  - [x] A4: Self-Attention
  - [x] A5: Combined

- [x] **Experiment 2**: Attention-Position Ablation (3 models)
  - [x] B2: Channel+Spatial at P2
  - [x] B3: Channel+Spatial at P3
  - [x] B4: Channel+Spatial at P4

- [ ] **Experiment 3**: Multiple Attention Positions
  - [ ] P3 only
  - [ ] P4 only
  - [ ] P3 + P4
  - [ ] P2 + P3 + P4 (extended)

- [ ] **Experiment 4**: Attention Order
  - [ ] Channel → Spatial
  - [ ] Spatial → Channel

- [ ] **Experiment 5**: Spatial Kernel Size
  - [ ] 3×3
  - [ ] 5×5
  - [ ] 7×7

- [ ] **Extended**: Additional Mechanisms
  - [ ] SE Attention (minimum 2 additional)
  - [ ] ECA (optional)
  - [ ] CBAM (optional)

- [x] **Grad-CAM**: Visual Explanation
  - [x] Baseline vs Attention comparison

- [ ] **Attention Maps**: Position visualization
  - [ ] P2, P3, P4 comparison

- [ ] **Graphical Comparison**: Performance visualization
  - [ ] At least 3 of 6 required plots

---

## 📈 REQUIRED RESULT TABLES

### Table 1: Main Comprehensive Results (Section 23)
| Model | Position | Accuracy | Precision | Recall | F1 | Parameters | Inference Time |
|-------|----------|----------|-----------|--------|----|-----------|----|
| Baseline | – | | | | | | |
| Channel | P3 | | | | | | |
| Spatial | P3 | | | | | | |
| Channel+Spatial | P2 | | | | | | |
| Channel+Spatial | P3 | | | | | | |
| Channel+Spatial | P4 | | | | | | |
| Self-Attention | P3 | | | | | | |
| Additional-1 | P3 | | | | | | |
| Additional-2 | P3 | | | | | | |

### Table 2: Attention-Type Ablation (All at P3)
| Model | Channel | Spatial | Self-Attn | Accuracy | Macro F1 | Parameters | Inference Time |
|-------|---------|---------|-----------|----------|----------|-----------|---|
| A0: Baseline | – | – | – | | | | |
| A1: Channel | ✓ | – | – | | | | |
| A2: Spatial | – | ✓ | – | | | | |
| A3: Ch+Sp | ✓ | ✓ | – | | | | |
| A4: Self-Attention | – | – | ✓ | | | | |
| A5: Combined | ✓ | ✓ | ✓ | | | | |

### Table 3: Position Ablation (Channel→Spatial)
| Position | Resolution | Channels | Accuracy | Macro F1 | Parameters | Inference Time |
|----------|-----------|----------|----------|----------|-----------|---|
| P2 | 28×28 | 512 | | | | |
| P3 | 14×14 | 1024 | | | | |
| P4 | 7×7 | 2048 | | | | |

### Table 4: Multiple Positions
| Config | Accuracy | Macro F1 | Parameters | Inference Time | Training Time |
|--------|----------|----------|-----------|---|---|
| P3 only | | | | | |
| P4 only | | | | | |
| P3 + P4 | | | | | |
| P2 + P3 + P4 | | | | | |

### Table 5: Attention Order
| Model | Order | Accuracy | Macro F1 | Parameters | Inference Time |
|-------|-------|----------|----------|-----------|---|
| Channel+Spatial | Ch→Sp | | | | |
| Channel+Spatial | Sp→Ch | | | | |

### Table 6: Kernel Size
| Kernel Size | Accuracy | Macro F1 | Inference Time |
|------------|----------|----------|---|
| 3×3 | | | |
| 5×5 | | | |
| 7×7 | | | |

### Table 7: Mechanism Comparison
| Attention | Accuracy | Precision | Recall | Macro F1 | Parameters | Inference Time |
|-----------|----------|-----------|--------|----------|-----------|---|
| No Attention | | | | | | |
| Channel/SE | | | | | | |
| Spatial | | | | | | |
| ECA | | | | | | |
| CBAM | | | | | | |
| Self-Attention | | | | | | |
| Additional | | | | | | |

---

## 📊 REQUIRED PLOTS & VISUALIZATIONS

### Main Plots (At least 3 of these 6)
1. **Training & Validation Accuracy** - Epoch vs Accuracy (train/val)
2. **Training & Validation Loss** - Epoch vs Loss (train/val)
3. **Mechanism vs Accuracy** - Bar graph of all mechanisms
4. **Mechanism vs Macro F1** - Bar graph of F1 scores
5. **Accuracy-Complexity Trade-off** - Scatter (Parameters vs Accuracy)
6. **F1-Inference Trade-off** - Scatter (Inference Time vs F1)

### Additional Visualizations (14 required total)
7. Sample input images from all 5 classes
8. Training accuracy curves comparison
9. Validation accuracy curves comparison
10. Training loss curves comparison
11. Confusion matrix (baseline & attention model)
12. Feature map BEFORE attention
13. Channel attention weights (bar graph)
14. Spatial attention heatmap
15. Spatial attention overlay on input
16. Self-attention response map
17. Feature map AFTER attention
18. Baseline Grad-CAM
19. Grad-CAM for attention model
20. Attention maps at P2, P3, P4
21. Per-class accuracy comparison
22. Parameter count comparison

---

## 🎯 KEY CALCULATIONS & FORMULAS

### Channel Attention
```
z_c = (1/HW) Σ_i Σ_j F(i,j,c)           [Global average pooling]
M_c(F) = σ(W_2 δ(W_1 z))               [Learnable transformation]
F_c = F ⊙ M_c(F)                        [Element-wise multiplication]

Where:
- δ = ReLU activation
- σ = Sigmoid activation  
- W_1: reduces C to C/16
- W_2: restores C/16 to C
- r = 16 (reduction ratio)
```

### Spatial Attention
```
F_avg = Mean_c(F)                      [Average pooling across C]
F_max = Max_c(F)                       [Max pooling across C]
F_s = [F_avg ∥ F_max]                 [Concatenate to H×W×2]
M_s(F) = σ(Conv_7×7(F_s))             [7×7 Conv + Sigmoid]
F'_s = F ⊙ M_s(F)                     [Element-wise multiplication]
```

### Sequential Channel-Spatial
```
F_c = M_c(F) ⊙ F                       [Channel attention first]
F_cs = M_s(F_c) ⊙ F_c                 [Spatial attention second]
```

### Self-Attention
```
X = reshape(F) to (N, C) where N=HW   [Flatten spatial dims]
Q = XW_Q, K = XW_K, V = XW_V          [Query, Key, Value]
A = softmax(QK^T / √d)                [Attention matrix (N×N)]
Y = AV                                 [Apply attention to values]
F_SA = F + γ reshape(Y)                [Residual with learnable γ]
```

### Loss Function
```
L_CE = -(1/N) Σ_i Σ_c y_i,c log(ŷ_i,c)

Macro F1 = (1/C) Σ_c F1_c where:
F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)
```

### Grad-CAM
```
α_c_k = (1/HW) Σ_i Σ_j ∂y_c/∂A^k_ij  [Importance coefficients]
L_Grad-CAM = ReLU(Σ_k α_c_k A^k)     [Class activation map]
```

---

## ❓ CRITICAL DISCUSSION QUESTIONS (35 Total)

### ResNet50 Understanding
1. What is the purpose of residual connections?
2. What does conv2_x represent compared to conv5_x?
3. Why does spatial resolution decrease with depth?
4. Why do feature channels increase with depth?
5. What is meant by low-level, intermediate, high-level features?

### Attention Fundamentals
6. What is the purpose of attention mechanisms?
7. What does channel attention attempt to learn?
8. What does spatial attention attempt to learn?
9. How is channel attention different from spatial attention?
10. Why can channel and spatial attention be used sequentially?

### Attention Position Effects
11. Why might channel attention behave differently at conv3_x vs conv5_x?
12. Why is spatial attention useful at intermediate stages?
13. Why is self-attention computationally expensive at high resolution?
14. What happens to attention matrix size when spatial resolution increases?
15. How does attention performance vary with insertion position?

### Visualization & Interpretation
16. How does attention at P2 differ visually from P4?
17. Does multi-position attention always improve performance?
18. Does Channel→Spatial differ from Spatial→Channel?
19. How does spatial kernel size influence the attention map?
20. How does SE attention differ from ECA attention?

### Attention Mechanisms
21. How does CBAM combine channel and spatial attention?
22. How does self-attention differ from channel attention?
23. Does more parameters necessarily mean better accuracy?

### Grad-CAM vs Attention Maps
24. What is the difference between internal attention maps and Grad-CAM?
25. Why is Grad-CAM called a post-hoc explanation?

### Attention Effectiveness
26. Does attention suppress irrelevant backgrounds?
27. Are attention maps different for correctly vs incorrectly classified images?
28. Which flower classes benefit most from attention?

### Performance Analysis
29. What information does confusion matrix provide beyond accuracy?
30. Why is macro F1 useful in addition to accuracy?
31. What is meant by accuracy-complexity trade-off?
32. What is meant by F1-inference-time trade-off?

### Experimental Protocol
33. Why must every ablation use the same dataset split?
34. Why maintain the same training protocol for all comparisons?

### Conclusions
35. What conclusions can be drawn from Grad-CAM comparisons?

---

## 🔑 KEY INSIGHTS

### Attention Effectiveness Formula
```
Effectiveness = f(Type, Position, Resolution, Semantics, Complexity)
```

### Complete Analysis Requirements
```
Complete Analysis = Quantitative + Computational + Qualitative

Where:
- Quantitative: Accuracy, Precision, Recall, F1
- Computational: Parameters, Inference time
- Qualitative: Visualizations, Grad-CAM, attention maps
```

### Feature Hierarchy
- **P1/Early (56×56, 64 ch)**: Edges, colors, textures
- **P2/Early-Mid (28×28, 512 ch)**: Local patterns, flower parts
- **P3/Mid (14×14, 1024 ch)**: Object structures
- **P4/Late (7×7, 2048 ch)**: High-level semantic info

### Expected Observations
- Intermediate positions (P3) often optimal
- Sequential channel→spatial often better than reverse
- Self-attention: powerful but expensive
- Attention at multiple positions: diminishing returns
- Different flowers may require different positions

---

## 📋 CHECKLIST SUMMARY

### Pre-Training
- [x] Dataset loading and exploration
- [x] Image preprocessing (224×224×3)
- [x] ImageNet normalization
- [x] 70-15-15 stratified split
- [x] Data augmentation (5 techniques)
- [x] ResNet50 analysis

### Implementation
- [x] Channel Attention
- [x] Spatial Attention
- [x] Sequential Channel-Spatial
- [x] Self-Attention
- [x] SE Attention
- [ ] ECA Attention
- [ ] CBAM Implementation

### Training
- [x] Experiment 1: Attention-Type (6 models)
- [x] Experiment 2: Attention-Position (3 models)
- [ ] Experiment 3: Multiple Positions
- [ ] Experiment 4: Attention Order
- [ ] Experiment 5: Kernel Size

### Evaluation
- [ ] Accuracy, Precision, Recall, F1 per class
- [ ] Confusion matrices
- [ ] Parameter counts
- [ ] Inference times

### Visualizations
- [ ] Plot 1-6 (Main plots)
- [ ] Visualization 1-14 (All required)
- [ ] Grad-CAM maps
- [ ] Attention position maps

### Report
- [ ] All 18 sections
- [ ] All tables (1-7)
- [ ] All plots (1-14+)
- [ ] Discussion of all 35 questions

---

## 🎓 EXPECTED OUTCOMES

After completing this lab, students will:

1. ✓ Understand ResNet50 architecture and feature hierarchy
2. ✓ Implement 4+ attention mechanisms from scratch
3. ✓ Conduct controlled ablation studies
4. ✓ Analyze attention effectiveness through multiple lenses
5. ✓ Create professional visualizations and reports
6. ✓ Distinguish between different attention positions
7. ✓ Understand accuracy-complexity trade-offs
8. ✓ Interpret Grad-CAM visualizations
9. ✓ Recognize that attention effectiveness depends on context
10. ✓ Perform comprehensive deep learning research

---

## 📞 TROUBLESHOOTING

### Memory Issues
- Reduce batch size (16 or 8)
- Use gradient checkpointing
- Train on smaller dataset subset first

### Slow Training
- Use GPU if available
- Reduce image resolution initially
- Use mixed precision training

### Poor Attention Results
- Check that same augmentation is used for ALL models
- Verify exact same data splits across experiments
- Ensure proper attention insertion points
- Use stratified split for balanced representation

---

**Document Version**: 1.0  
**Created**: 2026  
**Lab Course**: CS3807 Deep Learning Laboratory  
**Institution**: Shiv Nadar University Chennai


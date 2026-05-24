# Advanced MRI Preprocessing Pipeline — Model 3

## Project Context

This project belongs to the Deep Learning final project for brain tumor classification using MRI images.

The repository already contains:
- `model1`: baseline CNN
- `model2`: deeper CNN / normalization improvements
- `model3`: advanced architecture (this implementation)

The project constraints explicitly forbid:
- pretrained models,
- transfer learning,
- external learned weights.

Therefore, the contribution of `model3` will focus on:

1. Advanced preprocessing
2. Multimodal structural representation
3. Residual learning
4. Attention mechanisms

This document defines the complete preprocessing design for `model3`.

---

# 1. Motivation

The baseline preprocessing already includes:
- resize,
- RGB conversion,
- normalization,
- augmentation,
- class balancing.

However, MRI classification presents additional challenges:

- low local contrast,
- diffuse tumor boundaries,
- heterogeneous image acquisition,
- noisy anatomical structures,
- high intra-class variability.

Traditional preprocessing improves optimization stability, but it does not explicitly enhance:
- structural information,
- spatial gradients,
- tumor boundaries,
- local anatomical contrast.

The goal of the advanced preprocessing pipeline is therefore:

> Enrich the spatial and structural representation of MRI images before they are processed by the CNN.

---

# 2. Design Philosophy

The preprocessing pipeline is based on the concept of:

# Multimodal Structural Representation Learning

Instead of feeding only RGB intensity information into the CNN, the model will receive multiple complementary representations of the same MRI image.

These representations include:
- original RGB intensities,
- local contrast enhancement,
- edge/gradient information.

This approach is inspired by:
- classical computer vision,
- medical image enhancement,
- biologically-inspired visual processing.

---

# 3. Biological Inspiration

The design is conceptually inspired by the work of:
- Hubel & Wiesel (1959)

Their experiments demonstrated that neurons in the primary visual cortex respond selectively to:
- edges,
- orientations,
- lines,
- local spatial patterns.

This idea later inspired:
- Neocognitron,
- convolutional neural networks,
- hierarchical feature extraction.

The preprocessing pipeline explicitly incorporates:
- edge detection,
- local contrast enhancement,
- spatial structure extraction.

Thus, part of the visual feature extraction process is shifted from purely learned CNN filters into engineered structural preprocessing.

---

# 4. Pipeline Overview

```text
MRI Image
→ RGB Conversion
→ Resize (224x224)
→ CLAHE Enhancement
→ Sobel Edge Extraction
→ Multi-Channel Fusion
→ Augmentation
→ Tensor Conversion
→ Channel-wise Normalization
```

Validation and test pipelines exclude augmentation.

---

# 4.1 Pipeline Goals

The advanced preprocessing pipeline has four main goals:

1. Improve local anatomical contrast.
2. Explicitly expose structural information.
3. Provide multimodal visual representations.
4. Improve CNN feature extraction quality.

Unlike traditional preprocessing pipelines that focus mainly on normalization and augmentation, this pipeline attempts to enrich the input representation itself.

---

# 4.2 Processing Order Rationale

The order of operations is extremely important.

The pipeline intentionally follows:

```text
CLAHE → Sobel → Fusion → Augmentation
```

because:

- CLAHE enhances local contrast before edge extraction.
- Sobel benefits from stronger local gradients.
- Fusion must occur before augmentation to preserve channel alignment.
- Augmentation must affect all channels identically.

Changing this order could:
- weaken edge quality,
- introduce spatial inconsistencies,
- reduce the usefulness of multimodal representations.

---

# 4.3 RGB Conversion

MRI images in the dataset are not fully standardized.
Some images may contain:
- grayscale formats,
- inconsistent channels,
- different encodings.

Therefore, the first preprocessing step ensures that every image is converted into a consistent RGB representation.

This guarantees:
- consistent tensor dimensions,
- compatibility with convolutional layers,
- deterministic preprocessing.

Implementation responsibility:

```python
_ensure_rgb(image)
```

This logic already exists in `model1` and should be reused.

---

# 4.4 Resize

The dataset contains highly heterogeneous image resolutions.

Observed resolutions range from:
- very small MRI slices,
- to high-resolution scans.

A fixed input size is required because CNN architectures require:
- consistent tensor dimensions,
- deterministic convolution shapes,
- stable batching.

The selected resolution is:

```python
(224, 224)
```

Reasons:
- preserves sufficient anatomical detail,
- computationally manageable,
- already used in previous models,
- simplifies architecture compatibility.

Implementation responsibility:

```python
cv2.resize(...)
```
or torchvision resize transforms.

This logic already exists in `model1` and should be generalized.

---

# 4.5 CLAHE Enhancement

## Motivation

MRI images frequently suffer from:
- low local contrast,
- weak tissue boundaries,
- low visibility tumor regions.

Traditional histogram equalization often introduces:
- excessive noise amplification,
- overexposed regions,
- loss of local structure.

CLAHE solves these problems.

---

## What is CLAHE?

CLAHE stands for:

# Contrast Limited Adaptive Histogram Equalization

Instead of equalizing the entire image globally, CLAHE:
- divides the image into local regions,
- performs local histogram equalization,
- limits excessive amplification.

This produces:
- stronger local contrast,
- improved anatomical visibility,
- better tumor boundary separation.

---

## Why CLAHE Helps MRI Classification

Tumors are localized structures.

Therefore:
- local contrast matters more than global contrast.

CLAHE improves:
- tissue separability,
- visibility of diffuse boundaries,
- local spatial gradients.

This directly benefits convolutional feature extraction.

---

## CLAHE Processing Flow

```text
RGB Image
→ Grayscale Conversion
→ CLAHE
→ Enhanced Contrast Channel
```

---

## File Responsibility

Create:

```text
models/common/preprocessing/clahe.py
```

Expected responsibilities:
- grayscale conversion,
- CLAHE configuration,
- contrast enhancement,
- returning enhanced channel.

---

## Suggested Parameters

```python
clipLimit=2.0
 tileGridSize=(8, 8)
```

Parameters should remain configurable.

---

# 4.6 Sobel Edge Extraction

## Motivation

Many MRI tumor characteristics depend heavily on:
- boundaries,
- contours,
- structural transitions,
- spatial gradients.

This is particularly important for:
- diffuse gliomas,
- infiltrative structures,
- low-contrast tumors.

---

## What is Sobel?

The Sobel operator approximates spatial derivatives:

\[
\frac{\partial I}{\partial x}
\]

and:

\[
\frac{\partial I}{\partial y}
\]

through convolution kernels.

---

## Sobel Kernels

Horizontal:

\[
G_x=
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
\]

Vertical:

\[
G_y=
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
\]

---

## Edge Magnitude

Final edge intensity:

\[
G=\sqrt{G_x^2+G_y^2}
\]

---

## Why Sobel Helps CNNs

CNNs naturally learn low-level edge detectors in early layers.

However:
- the dataset is relatively small,
- MRI edges are noisy,
- boundaries can be weak.

Providing explicit structural gradients:
- simplifies feature extraction,
- strengthens contour representation,
- improves anatomical structure visibility.

---

## Biological Inspiration

This preprocessing stage is directly inspired by:
- Hubel & Wiesel,
- orientation-sensitive neurons,
- early visual cortex processing.

The preprocessing therefore introduces:
# biologically-inspired edge representation.

---

## File Responsibility

Create:

```text
models/common/preprocessing/sobel.py
```

Expected responsibilities:
- Sobel X computation,
- Sobel Y computation,
- edge magnitude computation,
- normalization of edge channel.

---

# 4.7 Multi-Channel Fusion

## Motivation

Different image representations contain complementary information.

| Representation | Information Type |
|---|---|
| RGB | Texture / intensity |
| CLAHE | Local contrast |
| Sobel | Structural gradients |

The CNN should receive all of them simultaneously.

---

## Final Tensor Representation

The final tensor becomes:

```python
(5, H, W)
```

Channel layout:

| Channel | Description |
|---|---|
| 0 | R |
| 1 | G |
| 2 | B |
| 3 | CLAHE |
| 4 | Sobel |

---

## Why This Matters

This creates:

# multimodal structural representation learning

instead of simple RGB classification.

The CNN now receives:
- anatomical texture,
- local contrast enhancement,
- explicit edge information.

---

## File Responsibility

Create:

```text
models/common/preprocessing/fusion.py
```

Expected responsibilities:
- channel stacking,
- tensor alignment,
- datatype consistency,
- multimodal fusion.

---

# 4.8 Augmentation Strategy

## Important Constraint

Augmentation must occur:

# AFTER channel fusion

---

## Why?

All channels must remain spatially aligned.

Incorrect example:
- rotating RGB,
- but not rotating Sobel.

This would break:
- anatomical consistency,
- multimodal alignment,
- spatial correspondence.

---

## Training Augmentations

Apply only during training:

- RandomHorizontalFlip
- Small rotations
- Brightness/contrast variation

Suggested rotation range:

```python
±10° to ±15°
```

---

## Validation and Test

Validation and test datasets must remain deterministic.

Therefore:
- no random augmentation,
- only fixed preprocessing.

---

# 4.9 Tensor Conversion

After preprocessing and fusion:
- NumPy arrays are converted into PyTorch tensors.

Expected shape:

```python
(5, 224, 224)
```

This tensor becomes the direct input to:
- the residual CNN,
- attention modules,
- later feature extraction stages.

---

# 4.10 Channel-wise Normalization

## Important Difference from Model1

ImageNet normalization can no longer be reused because:
- the input is no longer RGB-only,
- the statistical distribution changed,
- new structural channels were introduced.

---

## Correct Strategy

Compute:

\[
\mu_c, \sigma_c
\]

for all channels:

\[
c \in \{R,G,B,CLAHE,Sobel\}
\]

---

## Why This Matters

Normalization:
- stabilizes gradients,
- improves convergence,
- prevents exploding activations,
- standardizes multimodal distributions.

---

## File Responsibility

Create:

```text
models/common/preprocessing/normalization.py
```

Expected responsibilities:
- per-channel statistics,
- normalization utilities,
- configurable mean/std support.

---

# 4.11 Expected Final Pipeline

## Training Pipeline

```text
MRI
→ RGB Conversion
→ Resize
→ CLAHE
→ Sobel
→ Multi-Channel Fusion
→ Augmentation
→ Tensor Conversion
→ Channel-wise Normalization
```

---

## Validation/Test Pipeline

```text
MRI
→ RGB Conversion
→ Resize
→ CLAHE
→ Sobel
→ Multi-Channel Fusion
→ Tensor Conversion
→ Channel-wise Normalization
```

No augmentation is applied during evaluation.

---

# 5. Refactoring Strategy

The implementation of `model3` must avoid:
- duplicated preprocessing logic,
- duplicated dataset code,
- duplicated utility functions,
- breaking the reproducibility of `model1`.

The objective is to transform the current preprocessing implementation into:
- a reusable,
- modular,
- extensible infrastructure.

---

# 5.1 Refactoring Philosophy

The preprocessing system should follow these principles:

| Principle | Description |
|---|---|
| Reusability | Shared logic should exist only once |
| Modularity | Each preprocessing operation should be isolated |
| Extensibility | New preprocessing strategies should be easy to add |
| Reproducibility | Model1 behavior must remain unchanged |
| Separation of Concerns | Dataset loading and preprocessing should remain independent |

---

# 5.2 Reusable Infrastructure

The following components from `model1` should be generalized and reused:

## Reuse Completely

- RGB conversion helpers
- Resize logic
- Dataset split logic
- Seed handling
- Metrics
- Confusion matrix utilities
- Early stopping
- Plotting utilities
- Device handling

## Refactor and Generalize

- dataset.py
- transforms pipeline
- normalization pipeline

These components must support configurable preprocessing strategies.

---

# 5.3 Final Folder Structure

```text
models/
│
├── common/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── base_preprocessing.py
│   │   ├── advanced_mri_preprocessing.py
│   │   ├── clahe.py
│   │   ├── sobel.py
│   │   ├── fusion.py
│   │   └── normalization.py
│   │
│   ├── dataset.py
│   ├── metrics.py
│   ├── utils.py
│   └── visualization.py
│
├── model1/
├── model2/
└── model3/
```

---

# 5.4 BasePreprocessing

Create:

```python
class BasePreprocessing:
```

Purpose:
- preserve the preprocessing behavior already used by model1,
- centralize reusable preprocessing logic,
- avoid code duplication.

Responsibilities:
- RGB conversion,
- resize,
- standard augmentation,
- tensor conversion,
- standard normalization.

Expected interface:

```python
class BasePreprocessing:
    def __call__(self, image, train=True):
        ...
```

---

# 5.5 AdvancedMRIPreprocessing

Create:

```python
class AdvancedMRIPreprocessing:
```

Purpose:
- implement the advanced multimodal MRI preprocessing pipeline,
- integrate structural enhancement,
- integrate edge extraction,
- prepare tensors for model3.

Responsibilities:
- CLAHE enhancement,
- Sobel extraction,
- multimodal fusion,
- augmentation,
- custom normalization.

Expected flow:

```text
MRI
→ RGB Conversion
→ Resize
→ CLAHE
→ Sobel
→ Multi-Channel Fusion
→ Augmentation
→ Tensor Conversion
→ Channel-wise Normalization
```

---

# 5.6 Dataset Refactor

The dataset loader must support preprocessing strategy injection.

Example:

```python
train_dataset = MRIDataset(
    ...,
    preprocessing_strategy=AdvancedMRIPreprocessing()
)
```

Model1 should continue using:

```python
BasePreprocessing()
```

Model3 should use:

```python
AdvancedMRIPreprocessing()
```

The dataset loader must NOT hardcode preprocessing operations.

---

# 5.7 Implementation Order

The implementation should be completed in the following order:

## Step 1
Create:

```text
models/common/preprocessing/
```

---

## Step 2
Move reusable preprocessing logic from `model1` into:

- `common/preprocessing/base_preprocessing.py`
- `common/dataset.py`

without breaking model1.

---

## Step 3
Implement:

```text
clahe.py
```

including:
- grayscale conversion,
- CLAHE processing,
- configurable parameters.

---

## Step 4
Implement:

```text
sobel.py
```

including:
- Sobel X,
- Sobel Y,
- Sobel magnitude,
- edge normalization.

---

## Step 5
Implement:

```text
fusion.py
```

including:
- channel stacking,
- multimodal tensor generation,
- datatype consistency.

---

## Step 6
Implement:

```text
normalization.py
```

including:
- configurable normalization,
- per-channel statistics,
- support for 5-channel tensors.

---

## Step 7
Implement:

```text
advanced_mri_preprocessing.py
```

integrating the complete preprocessing pipeline.

---

## Step 8
Refactor dataset loading to support:

```python
preprocessing_strategy=
```

---

## Step 9
Validate:
- tensor shapes,
- channel ordering,
- augmentation consistency,
- normalization ranges.

---

# 5.8 Validation Requirements

Before integrating model3, verify:

## Shape Validation

Expected final tensor:

```python
(5, 224, 224)
```

---

## Channel Validation

Verify:

| Channel | Expected Content |
|---|---|
| 0 | R |
| 1 | G |
| 2 | B |
| 3 | CLAHE |
| 4 | Sobel |

---

## Visualization Validation

Generate debug visualizations for:
- original MRI,
- CLAHE image,
- Sobel edges,
- fused tensor channels.

This is critical to ensure:
- preprocessing correctness,
- structural consistency,
- edge quality.

---

## Statistical Validation

Compute:
- per-channel mean,
- per-channel standard deviation,
- min/max ranges.

This is necessary before defining final normalization constants.

---

# 5.9 Engineering Constraints

The implementation must:

- avoid duplicated code,
- preserve model1 reproducibility,
- remain modular,
- support future preprocessing extensions,
- support future experiments.

The preprocessing system should be extensible enough to support future additions such as:
- Laplacian filters,
- Gabor filters,
- denoising,
- frequency-domain preprocessing,
- segmentation masks.

---

# 5.10 Expected Contribution of Model3

Model3 is expected to contribute:

## Structural MRI Enhancement
through:
- CLAHE,
- Sobel gradients.

## Multimodal Representation Learning
through:
- RGB + contrast + edge fusion.

## Bio-Inspired Processing
through:
- edge-oriented preprocessing,
- orientation-sensitive representations.

## Improved CNN Feature Extraction
through:
- stronger structural representations,
- enhanced local contrast,
- explicit spatial gradients.


This preprocessing pipeline should make model3 significantly different from:
- the baseline CNN,
- and the deeper CNN from model2.

---

# 6. Model3 Architecture Design

After completing the advanced preprocessing pipeline, the next stage is implementing the architecture for `model3`.

The goal of model3 is NOT simply:
- increasing depth,
- stacking more convolutions,
- or creating a larger CNN.

Instead, model3 is designed to address two major limitations of classical CNNs:

1. Gradient degradation in deeper networks.
2. Lack of adaptive feature prioritization.

To solve these problems, model3 integrates:
- residual learning,
- attention mechanisms,
- multimodal MRI preprocessing.

This creates a significantly more advanced architecture than:
- the baseline CNN,
- and the deeper CNN from model2.

---

# 6.1 Conceptual Architecture

The final conceptual pipeline becomes:

```text
MRI
→ Advanced MRI Preprocessing
→ Residual CNN Backbone
→ Attention Modules
→ Global Average Pooling
→ Dense Classification Head
→ Softmax Classification
```

---

# 6.2 Main Architectural Goals

The architecture has five major goals:

| Goal | Description |
|---|---|
| Stable Deep Training | Residual connections improve gradient flow |
| Better Structural Representation | Advanced preprocessing enriches MRI structure |
| Adaptive Feature Prioritization | Attention focuses on important features |
| Reduced Overfitting | GAP + residual design reduce parameter explosion |
| Better Generalization | Combination of preprocessing and architecture improves robustness |

---

# 6.3 Why Residual Learning?

Classical deep CNNs suffer from:
- vanishing gradients,
- degradation problems,
- unstable optimization.

As depth increases:
- gradients become smaller,
- early layers stop learning effectively,
- optimization becomes increasingly difficult.

Residual learning solves this problem.

---

# 6.4 Residual Learning Theory

Instead of learning:

\[
H(x)
\]

a residual block learns:

\[
F(x) + x
\]

where:
- \(x\) is the original input,
- \(F(x)\) is the learned residual transformation.

---

# 6.5 Intuition Behind Residual Learning

Residual connections allow the network to:
- preserve useful information,
- improve gradient propagation,
- stabilize optimization,
- train deeper architectures.

Instead of forcing every layer to completely transform the representation, the network learns only:
- the residual correction.

This greatly simplifies optimization.

---

# 6.6 Residual Block Structure

The proposed residual block follows the structure:

```text
Input
│
├── Conv2D
├── BatchNorm
├── ReLU
├── Conv2D
├── BatchNorm
│
└── Skip Connection (+)
        ↓
      ReLU
```

---

# 6.7 Residual Block Responsibilities

Residual blocks are responsible for:
- hierarchical feature extraction,
- stable deep learning,
- preserving spatial information,
- improving gradient flow.

---

# 6.8 Why Residual Learning Fits MRI

MRI classification depends heavily on:
- subtle spatial patterns,
- small structural variations,
- weak boundaries.

Residual learning helps preserve:
- low-level anatomical information,
- structural gradients,
- local contrast patterns.

This is especially important because the preprocessing pipeline already introduces:
- edge-enhanced representations,
- multimodal structural channels.

---

# 6.9 Attention Mechanisms

Classical CNNs process all features with approximately equal importance.

However:
- not all channels are equally informative,
- not all spatial regions contain tumors,
- some features represent noise.

Attention mechanisms solve this problem.

---

# 6.10 Goal of Attention

Attention mechanisms allow the network to learn:

# what features are important

and:

# what features should be suppressed.

This creates adaptive feature prioritization.

---

# 6.11 Selected Attention Mechanism

The recommended attention module is:

# Squeeze-and-Excitation (SE) Attention

because it is:
- lightweight,
- computationally efficient,
- easy to integrate,
- highly effective for CNNs.

---

# 6.12 SE Block Theory

SE blocks operate in three stages:

| Stage | Purpose |
|---|---|
| Squeeze | Aggregate global channel information |
| Excitation | Learn channel importance weights |
| Reweighting | Amplify or suppress channels |

---

# 6.13 Squeeze Stage

Global Average Pooling compresses spatial information:

\[
z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j)
\]

This produces:
- one descriptor per channel.

---

# 6.14 Excitation Stage

The descriptor vector is passed through:
- fully connected layers,
- nonlinear activations,
- sigmoid scaling.

This produces channel weights:

\[
s_c \in [0,1]
\]

representing feature importance.

---

# 6.15 Reweighting Stage

Channels are reweighted:

\[
\hat{x}_c = s_c \cdot x_c
\]

Important channels are amplified.

Irrelevant channels are suppressed.

---

# 6.16 Why Attention Helps MRI

MRI images contain:
- irrelevant background regions,
- anatomical variability,
- noisy structures.

SE attention helps the network focus on:
- tumor boundaries,
- relevant anatomical patterns,
- diagnostically important structures.

This becomes especially powerful when combined with:
- Sobel edge channels,
- CLAHE-enhanced structures.

---

# 6.17 Global Average Pooling (GAP)

Instead of using a large flatten layer followed by massive dense layers, model3 will use:

# Global Average Pooling

---

# 6.18 Why GAP?

Model1 contains a very large dense layer:

```text
Flatten → Linear(50176, 128)
```

This introduces:
- millions of parameters,
- overfitting risk,
- high memory consumption.

GAP solves this problem.

---

# 6.19 GAP Theory

GAP computes:

\[
y_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j)
\]

for each channel.

This converts:

```text
(C, H, W)
```

into:

```text
(C)
```

without introducing large dense parameter matrices.

---

# 6.20 Benefits of GAP

| Benefit | Description |
|---|---|
| Fewer Parameters | Reduces overfitting |
| Better Generalization | Simpler classifier head |
| Spatial Robustness | Aggregates global spatial information |
| Better Stability | Avoids huge dense layers |

---

# 6.21 Proposed Model3 Architecture

The recommended architecture is:

```text
Input Tensor (5 Channels)
│
├── Initial Conv Block
│
├── Residual Block × N
│       ├── Conv
│       ├── BN
│       ├── ReLU
│       ├── Conv
│       ├── BN
│       └── Skip Connection
│
├── SE Attention Blocks
│
├── Global Average Pooling
│
├── Dense Layer
│
├── Dropout
│
└── Softmax Output (4 Classes)
```

---

# 6.22 Suggested Depth

Recommended architecture depth:

| Stage | Channels |
|---|---|
| Initial Conv | 32 |
| Residual Stage 1 | 32 |
| Residual Stage 2 | 64 |
| Residual Stage 3 | 128 |
| Final Features | 256 |

This provides:
- enough representational power,
- manageable computational cost,
- stable optimization.

---

# 6.23 Initial Convolution

Because preprocessing outputs:

```python
(5, 224, 224)
```

the first convolution must accept:

```python
Conv2d(5, 32, kernel_size=3, padding=1)
```

instead of:

```python
Conv2d(3, ...)
```

This is a critical architectural modification.

---

# 6.24 Batch Normalization

Batch Normalization should be used throughout model3.

Responsibilities:
- stabilize activations,
- reduce internal covariate shift,
- accelerate convergence,
- improve optimization stability.

BN is particularly important because:
- model3 uses deeper residual blocks,
- multimodal inputs introduce heterogeneous distributions.

---

# 6.25 Activation Function

Recommended activation:

# ReLU

Reasons:
- computationally efficient,
- avoids sigmoid saturation,
- stable optimization,
- standard for residual CNNs.

Optional future experiment:
- LeakyReLU,
- GELU.

---

# 6.26 Dropout Strategy

Dropout should be applied only in:
- the classifier head.

Recommended value:

```python
Dropout(0.5)
```

Avoid excessive dropout inside residual blocks because:
- it may disrupt residual information flow.

---

# 6.27 Weight Initialization

Use:

# Kaiming Initialization

for:
- convolution layers,
- linear layers.

Reason:
- optimized for ReLU activations,
- improves convergence stability.

---

# 6.28 Loss Function

Use:

# CrossEntropyLoss

with:

```python
class_weight
```

because the dataset contains class imbalance.

This preserves consistency with:
- model1,
- previous experiments.

---

# 6.29 Optimizer

Recommended optimizer:

# AdamW

because:
- adaptive learning,
- decoupled weight decay,
- stable optimization for deep CNNs.

Suggested configuration:

```python
lr=1e-4
weight_decay=1e-4
```

---

# 6.30 Learning Rate Scheduler

Recommended scheduler:

# Cosine Annealing

or:

# ReduceLROnPlateau

Purpose:
- stabilize late training,
- improve convergence,
- avoid oscillation.

---

# 6.31 Early Stopping

Early stopping should monitor:

# Validation Macro F1-Score

instead of only:
- validation accuracy.

Reason:
- the dataset is imbalanced,
- macro F1 better reflects class performance.

---

# 6.32 Main Evaluation Metric

Primary metric:

# Macro F1-Score

because it:
- treats all classes equally,
- penalizes minority class neglect,
- provides better medical classification evaluation.

---

# 6.33 Expected Advantages of Model3

Model3 is expected to improve:

| Improvement | Mechanism |
|---|---|
| Better edge representation | Sobel preprocessing |
| Better local contrast | CLAHE |
| Better deep optimization | Residual learning |
| Better feature prioritization | SE attention |
| Lower overfitting | GAP + reduced dense layers |
| Better generalization | Multimodal representation |

---

# 6.34 Expected Experimental Comparisons

The project should compare:

| Model | Main Characteristic |
|---|---|
| Model1 | Baseline CNN |
| Model2 | Deeper CNN + BN |
| Model3 | Residual + Attention + Multimodal MRI preprocessing |

---

# 6.35 Ablation Study Recommendations

To evaluate the contribution of each component, perform experiments such as:

| Experiment | Purpose |
|---|---|
| Without CLAHE | Evaluate contrast enhancement impact |
| Without Sobel | Evaluate edge information impact |
| Without Attention | Evaluate SE contribution |
| Without Residual Blocks | Evaluate residual learning |
| RGB-only | Compare against multimodal representation |

These experiments greatly strengthen the academic quality of the project.

---

# 6.36 Final Conceptual Pipeline

```text
MRI Dataset
│
├── Advanced MRI Preprocessing
│       ├── CLAHE
│       ├── Sobel
│       ├── Fusion
│       └── Normalization
│
├── Residual CNN Backbone
│
├── SE Attention Modules
│
├── Global Average Pooling
│
├── Dense Classification Head
│
└── Softmax Classification
```

---

# 6.37 Expected Contribution of Model3

Model3 contributes:

## Advanced MRI Representation
through:
- multimodal preprocessing,
- structural enhancement,
- explicit edge extraction.

## Stable Deep Learning
through:
- residual connections,
- Batch Normalization,
- optimized gradient flow.

## Adaptive Feature Learning
through:
- SE attention,
- channel reweighting,
- feature prioritization.

## Reduced Overfitting
through:
- Global Average Pooling,
- smaller classifier head,
- improved architectural efficiency.

This architecture should provide a substantially more advanced and research-oriented solution compared with the previous models.

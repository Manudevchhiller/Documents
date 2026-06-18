# Adaptive-Geometric-SMOTE (AGS): Research-Style Technical Note

## Abstract
Adaptive-Geometric-SMOTE (AGS) is a hybrid oversampling method designed to improve on classical and modern SMOTE variants by combining adaptive neighborhood sizing, manifold-aware geometric synthesis, and quality-checked sample selection. AGS explicitly addresses blind class overlap generation, boundary complexity, inefficiency of density-only methods, and weak high-dimensional behavior.

## Motivation and limitations addressed
AGS targets six recurring issues reported in prior methods:
1. Overlapping sample generation under blind interpolation
2. Poor adaptation to complex decision boundaries
3. Density-only sampling inefficiency
4. Runtime cost on larger data
5. Static parameter settings
6. Weak behavior in high-dimensional spaces

## Method summary
For each minority class:
1. Compute manifold projection (PCA) in high dimensions
2. Estimate local density and derive **adaptive per-point k**
3. Estimate boundary/safety by mixed-neighborhood composition
4. Estimate structure priority from minority mini-batch clustering
5. Fuse signals into generation probability
6. Generate candidates using line interpolation with truncated Gaussian coefficient
7. Validate each candidate (minority-majority distance ratio + local safety)
8. Keep validated samples only

## Computational design
- KD-tree accelerated nearest-neighbor queries
- Vectorized NumPy computations for score construction
- Optional CuPy backend hook for GPU vectorized operations
- MiniBatchKMeans for scalable clustering signal

## Evaluation protocol
Use the included notebook to compare AGS with baseline methods (e.g., RandomOverSampler/SMOTE variants available in your local environment).

Recommended metrics:
- F1-score
- AUC-ROC
- G-mean
- Sensitivity
- Specificity

Recommended settings:
- Binary and multi-class imbalanced datasets
- High-dimensional synthetic data
- Noisy and boundary-overlapping configurations

## Expected behavior
AGS is designed to improve boundary quality and minority recall while constraining overlap through acceptance filtering. In difficult noisy/boundary settings it may produce fewer but safer points due to conservative candidate rejection.

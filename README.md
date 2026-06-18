# Documents

## Adaptive-Geometric-SMOTE (AGS)

This repository now includes a **next-generation oversampling model** designed from the referenced papers:

- **Implementation:** `/home/runner/work/Documents/Documents/ags_smote.py`
- **Unit tests:** `/home/runner/work/Documents/Documents/tests/test_ags_smote.py`
- **Example benchmark notebook:** `/home/runner/work/Documents/Documents/examples/ags_benchmark.ipynb`
- **Research-style writeup:** `/home/runner/work/Documents/Documents/AGS_RESEARCH_NOTE.md`

### AGS highlights

- Adaptive local-k selection from minority density
- Dynamic sampling size via class separability
- Geometric generation with truncated Gaussian interpolation
- Multi-perspective scoring (density + clustering + boundary + safety)
- Quality validation and overlap/noise filtering before injection
- KD-tree indexing for efficient nearest-neighbor operations
- scikit-learn compatible API style (`fit_resample`)
- Optional GPU backend (uses CuPy if available)

### Quick start

```python
from ags_smote import AdaptiveGeometricSMOTE

ags = AdaptiveGeometricSMOTE(random_state=42)
X_res, y_res = ags.fit_resample(X, y)
```

### Local validation

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
```

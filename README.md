# Tensor Decompositions for OPM-MEG Analysis

A Python-based toolkit for applying tensor decomposition methods to Optically Pumped Magnetometer (OPM) Magnetoencephalography (MEG) data. This project implements CP/PARAFAC decomposition with rank selection algorithms for analyzing neural time-frequency data.

## Overview

This repository provides tools for decomposing multi-dimensional MEG data into interpretable components using tensor factorization methods. The analysis pipeline includes preprocessing, decomposition, rank selection, and visualization of neural activity patterns.

## Key Features

- **CP/PARAFAC Decomposition**: Implementation of canonical polyadic tensor decomposition with optional non-negativity constraints
- **Rank Selection Methods**:
  - Stability analysis across multiple decomposition runs
  - Model fit evaluation (R² scores)
  - CORCONDIA (Core Consistency Diagnostic)
- **MEG Preprocessing**: Time-frequency analysis, epoching, and filtering pipelines
- **Behavioral Analysis**: Win-Stay-Lose-Shift (WSLS) modeling and trial timing analysis
- **Visualization Tools**: t-SNE, evoked responses, and component plotting

## Structure

```
.
├── decompositions/     # Source decomposition notebooks (choice/outcome conditions)
├── preprocessing/      # MEG data preprocessing (filtering, epoching, time-frequency)
├── src/
│   ├── tools/         # Rank selection algorithms (rankselection.py)
│   └── utils/         # Plotting utilities
├── Behaviour/         # Behavioral data and WSLS analysis
├── visualisations/    # t-SNE and visualization notebooks
└── tests/            # Unit tests
```

## Dependencies

- PyTorch
- TensorLy
- MNE-Python (for MEG preprocessing)
- NumPy
- SciPy
- Matplotlib

## Usage

### Rank Selection

```python
from src.tools.rankselection import rank_selection, suggest_rank

# Run stability and fit analysis
results = rank_selection(
    tensor_data,
    ranks=range(1, 11),
    n_repeats=10,
    non_negative_modes=[1, 2]  # e.g., frequency and time modes
)

# Get suggested optimal rank
optimal_rank = suggest_rank(results, stability_threshold=0.85, fit_threshold=0.8)
```

### CP Decomposition

```python
from src.tools.rankselection import run_parafac

# Perform decomposition with constraints
cp_tensor = run_parafac(
    tensor_data,
    rank=5,
    non_negative_modes=[1, 2]
)
```

## Methods

The toolkit uses the Hungarian algorithm for optimal component matching and similarity scoring across decomposition runs. Stability is assessed by comparing factor matrices using absolute cosine similarity to handle sign ambiguity in CP decomposition.

## Data

This project analyzes OPM-MEG data collected during decision-making tasks. Behavioral data includes trial timing and choice outcomes for cognitive modeling.

## License

Research project - UCL Masters thesis

## Contact

For questions or collaboration, please open an issue on this repository.

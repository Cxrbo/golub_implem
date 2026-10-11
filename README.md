# ml-bio-golub

Reimplementation of the **Golub** algorithm (*weighted voting*, Golub et al., 1999) and comparison with a **logistic regression**, applied to the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset: classification of breast masses as *malignant* / *benign* from cell nuclei features.

## Project structure

```
ml_bio_golub/
├── data/data/
│   ├── wdbc.csv                   # 569 samples, 30 features + sample_id + diagnosis
│   ├── DATA_DICTIONARY.md         # Column descriptions, context and citations
│   └── MANIFEST.json              # Provenance (scikit-learn / UCI) and SHA-256 checksum
├── src/ml_bio_golub/
│   ├── __init__.py
│   ├── main.py                    # Entry point: runs the whole pipeline
│   ├── explore_data.py            # Descriptive statistics + class distribution plot
│   ├── weighted_voting.py         # weighted_voting_model class (Golub algorithm)
│   ├── evalution_of_model.py      # Weighted voting evaluation (cross-validation)
│   └── logistic_regression.py     # Logistic regression evaluation
├── pyproject.toml
├── uv.lock
└── .python-version                # Python 3.11
```

## Installation

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python ≥ 3.11 automatically if needed).

```bash
git clone <repository-url>
cd ml_bio_golub

uv sync
```

`uv sync` creates the `.venv` and installs the dependencies declared in `pyproject.toml` (pandas, numpy, scikit-learn, matplotlib).

## Usage

From the project root (the folder containing `data/` and `pyproject.toml`):

```bash
uv run ml-bio-golub                        # uses data/data/wdbc.csv by default
uv run ml-bio-golub path/to/other.csv      # or a custom CSV file
```

Results are written to a `results/` folder created in the current directory:

```
results/
├── data/
│   ├── malignant_vs_benign.png
│   └── reports_data.txt
├── weighted_voting/
│   ├── reports_weighted_voting.txt    # per-fold reports + averages
│   ├── avg_confusion_matrix.png
│   └── feature_frequency.png          # selection frequency of each feature
└── logistic_regression/
    ├── reports_logistic_regression.txt
    └── avg_confusion_matrix.png
```

## References

- Golub, T. R. et al. (1999). *Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring.* Science, 286(5439), 531–537.
- Wolberg, W. H., Street, W. N., Mangasarian, O. L. (1994). *Machine learning techniques to diagnose breast cancer from image-processed nuclear features of fine needle aspirates.* Cancer Letters, 77(2–3), 163–171. DOI: 10.1016/0304-3835(94)90099-X.
- Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE Electronic Imaging, 1905, 861–870. DOI: 10.1117/12.148698.
- Wolberg, W., Mangasarian, O., Street, N., Street, W. *Breast Cancer Wisconsin (Diagnostic).* UCI Machine Learning Repository. DOI: 10.24432/C5DW2B.

## Author

Arthaud Evrard (arthaud.evrard@epita.fr)
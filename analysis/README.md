# Reproducible analysis and internship report

This folder adds exploratory data analysis, a fixed Logistic Regression baseline,
evaluation of the existing MLP checkpoint, and descriptive test-error analysis.
It does not retrain or overwrite the supplied neural network.

## Run

From the repository root, using Python 3.11:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/analyze.py
python analysis/build_report.py
```

The PDF builder uses Arial on Windows or DejaVu Sans on Linux. If neither font is
available, supply a Unicode TrueType font with `--font /path/to/font.ttf`.
The optional `--output-dir` argument changes the report output directory.
The checked-in Markdown report expects the repository's `report/` directory so
its relative figure links resolve correctly.

## Outputs

- `results/metrics.json`: exact metrics, split checks, feature summaries,
  checkpoint hash, baseline settings, and analysis package versions.
- `results/test_errors.csv`: all 121 misclassified final-test rows from the MLP.
- `results/figures/eda_distribution.png`: class balance and training light distributions.
- `results/figures/eda_timeline.png`: training CO2 and observed occupancy over time.
- `results/figures/confusion_matrix.png`: saved MLP's final-test confusion matrix.
- `results/figures/error_scatter.png`: correct/incorrect predictions in two sensor dimensions.
- `../report/oda_doluluk_proje_raporu.pdf`: five-page Turkish internship report.
- `../report/oda_doluluk_proje_raporu.md`: editable report text and figure links.

## Method

The baseline uses StandardScaler and LogisticRegression with settings fixed before
baseline test evaluation: `C=1`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`.
Only the training split fits the scaler and classifier. No hyperparameter search
or test-based checkpoint selection is performed. The MLP uses its existing
checkpoint's training statistics and the same 0.5 threshold as the baseline.

All reported precision, recall, and F1 values use occupied (1) as the positive
class. The always-unoccupied baseline assigns zero to undefined precision via
`zero_division=0`. Dataset files retain their original contents and split roles.

Error proximity to a transition means an absolute time difference of at most
120 seconds from the first test record whose true label differs from its
predecessor. It is retrospective descriptive analysis using true labels, not an
inference feature. The example errors are the first three chronological examples
of each error type, rather than examples chosen by confidence.

The original checkpoint does not include its selected epoch or a training-loss
history. No training curve has been reconstructed or invented. After examining
these test errors, future tuning should use validation data and reserve a new
independent test set for the final assessment.

The tested analysis environment is recorded in `results/metrics.json`; dependency
minimums above do not guarantee byte-identical results on every environment.

## Report coverage

The report covers the supplied project guideline's problem statement, dataset
source and imbalance, three EDA views with interpretation, preprocessing,
architecture and rationale, training settings, evaluation with visual confusion
matrix, actual error analysis, limitations, development suggestions, and GitHub link.
Its author is Mert Ali Korkmaz and the internship institution is Software Persona.

Data attribution: Candanedo, L. (2016). *Occupancy Detection*. UCI Machine
Learning Repository. https://doi.org/10.24432/C5X01N. Dataset license: CC BY 4.0.

# Smart Room Occupancy

**Five sensor readings. One room. A small neural network that predicts whether it is occupied.**

A hands-on PyTorch learning project built around real office sensor measurements. It includes model training, validation-based early stopping, final-test evaluation, and an interactive terminal predictor with a pretrained checkpoint.

| Final-test accuracy | Final-test F1 | Trainable parameters |
| :---: | :---: | :---: |
| **98.76%** | **97.11%** | **241** |

These results describe the included checkpoint on the supplied test split, not guaranteed performance in other rooms.

## Quick start

Use Python 3.11 and run the following commands from the project directory. A virtual environment is recommended.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS / Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and start the predictor:

```bash
python -m pip install torch pandas
python predict.py
```

The predictor uses the included `occupancy_model.pth`; no training run or dataset files are needed for prediction. Its terminal messages are in Turkish. Training and evaluation additionally require pandas and the supplied data files. Dependencies are not currently version-pinned.

## Try a prediction

Enter the five measurements when prompted, using a dot for decimals:

```text
Temperature (°C): 22.5
Humidity (%): 28
Light (lux): 350
CO2 (ppm): 900
Humidity Ratio: 0.0045

Doluluk Olasılığı: %76.51
Tahmin: ODA DOLU
```

`ODA DOLU` means occupied; `ODA BOŞ` means unoccupied. A sigmoid output of at least `0.5` is classified as occupied. The displayed percentage is a model score, not a calibrated guarantee of actual occupancy.

Non-numeric input, NaN, infinity, and values outside the implemented physical bounds are rejected with a retry prompt. These bounds do not guarantee that a reading resembles the training data.

## Inputs and target

The feature order is identical during training, evaluation, and prediction:

| Order | Feature | Unit |
| --- | --- | --- |
| 1 | Temperature | °C |
| 2 | Humidity | Relative humidity, % |
| 3 | Light | lux |
| 4 | CO2 | ppm |
| 5 | HumidityRatio | kg water vapor / kg air |

The target is `Occupancy`: **0 = unoccupied**, **1 = occupied**. Timestamps and row IDs are not model inputs.

## Dataset

This project uses the [UCI Occupancy Detection dataset](https://archive.ics.uci.edu/dataset/357/occupancy+detection), contributed by Luis Candanedo. Ground-truth occupancy was obtained from timestamped photographs taken every minute.

| File | Role in this project | Samples | Measurement dates (2015) |
| --- | --- | ---: | --- |
| `data/datatraining.txt` | Training and scaling statistics | 8,143 | February 4–10 |
| `data/datatest.txt` | Validation and checkpoint selection | 2,665 | February 2–4 |
| `data/datatest2.txt` | Final evaluation | 9,752 | February 11–18 |

The supplied splits have no shared timestamps. Validation precedes the training period; final testing follows it. The final-test file is not read by the training script.

**Attribution:** Candanedo, L. (2016). *Occupancy Detection* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5X01N](https://doi.org/10.24432/C5X01N).

The dataset is distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The included data files are unchanged. This dataset license does not assign a license to the project's source code; no separate source-code license has been selected.

## Model and preprocessing

```text
5 sensor features
       ↓
Linear(5, 16) → ReLU
       ↓
Linear(16, 8) → ReLU
       ↓
Linear(8, 1)
       ↓
Raw logit → sigmoid for prediction
```

`OccupancyCalc` has **241 trainable parameters**. Training uses `BCEWithLogitsLoss`, so the model's forward pass returns a raw logit without a sigmoid layer.

Features are standardized as `(x - mean) / std`. Both statistics are computed only from the training data, with population standard deviation (`correction=0`). Zero standard deviations are replaced with `1`. Validation, test, and interactive prediction reuse those training statistics.

The small pretrained checkpoint is included for immediate use and contains:

```python
{
    "model_state_dict": ...,
    "mean": ...,
    "std": ...,
}
```

## Training

Run from the project directory:

```bash
python train.py
```

**This command overwrites `occupancy_model.pth`. Back up the included checkpoint before retraining if you want to retain the reported model.**

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early-stopping patience | 5 epochs |
| PyTorch seed | 42 |

Training batches are shuffled. Validation runs without gradient updates after each epoch. A validation loss no higher than the best recorded loss updates the saved weights and resets patience. The best weights are restored before saving, including when the maximum epoch count is reached.

The periodic `Loss` message is the last training batch's loss; the final `Train Loss` is calculated over the full training split. A seed is set, but identical results across package versions and hardware are not guaranteed.

## Evaluation

Run from the project directory:

```bash
python test.py
```

This evaluates the saved checkpoint on `datatest2.txt` and prints loss, accuracy, precision, recall, F1, and the confusion matrix. Precision, recall, and F1 are printed as fractions by the script; the table below expresses them as percentages.

### Included checkpoint results

| Metric | Validation | Final test |
| --- | ---: | ---: |
| Loss | 0.0770 | 0.0694 |
| Accuracy | 97.86% | 98.76% |
| Precision | 94.63% | 94.96% |
| Recall | 99.79% | 99.37% |
| F1 | 97.15% | 97.11% |

Full-training-split loss: **0.0426**.

Final-test confusion matrix (rows = actual class, columns = predicted class):

| | Predicted unoccupied | Predicted occupied |
| --- | ---: | ---: |
| Actual unoccupied | 7,595 | 108 |
| Actual occupied | 13 | 2,036 |

Matrix layout: `[[TN, FP], [FN, TP]]`.

These numbers were independently reproduced from the supplied checkpoint and CSV files using PyTorch. Retraining may produce different results. Keep the final-test split separate from hyperparameter tuning and checkpoint selection.

## Project layout

```text
smart-room-occupancy/
├── data/
│   ├── datatraining.txt
│   ├── datatest.txt
│   └── datatest2.txt
├── model.py               # Neural network definition
├── train.py               # Training, validation, early stopping, saving
├── test.py                # Final-test evaluation
├── predict.py             # Interactive sensor input and inference
├── occupancy_model.pth    # Pretrained weights and training statistics
└── README.md
```

## Limitations and current behavior

- Measurements come from one office setting over a limited period. Generalization to other buildings, sensor hardware, lighting patterns, and seasons has not been established.
- Occupied samples account for about 21% of training/final-test data and 36.5% of validation data. Consider precision, recall, and F1 alongside accuracy.
- This is a learning project, not a production occupancy or safety system. Probability calibration and out-of-distribution detection are not implemented.
- Run training and evaluation from the project directory because their file paths are relative. Prediction resolves its checkpoint path relative to its script.
- Training, evaluation, and prediction execute at module level. Use them as scripts; importing `train.py` starts training and writes a checkpoint.
- Evaluation metric divisions are not guarded against zero denominators. The included checkpoint evaluates successfully on the provided splits; degenerate predictions or different datasets may need additional handling.

Built with **Python**, **PyTorch**, and **pandas**.

"""Reproduce EDA, a fixed logistic baseline, and saved-model error analysis.

Run from the repository root: python analysis/analyze.py
Never trains or overwrites the supplied neural-network checkpoint.
"""
import csv
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import OccupancyCalc

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
OUT = ROOT / "analysis" / "results"
FIG = OUT / "figures"


def read_split(name):
    with (ROOT / "data" / (name + ".txt")).open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        # UCI files have an unlabeled row-ID field before the named columns.
        rows = [dict(zip(["id"] + header, row)) for row in reader]
    x = np.array([[float(row[k]) for k in FEATURES] for row in rows])
    y = np.array([int(row["Occupancy"]) for row in rows])
    dates = [datetime.fromisoformat(row["date"]) for row in rows]
    assert np.isfinite(x).all() and set(y) <= {0, 1}
    return rows, x, y, dates


def metrics(y, pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return dict(accuracy=accuracy_score(y, pred), precision=precision,
                recall=recall, f1=f1,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def save(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    checkpoint_path = ROOT / "occupancy_model.pth"
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    data = {key: read_split(name) for key, name in [
        ("train", "datatraining"), ("validation", "datatest"), ("test", "datatest2")]}
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model = OccupancyCalc()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x_train = torch.tensor(data["train"][1], dtype=torch.float32)
    assert torch.allclose(x_train.mean(0), checkpoint["mean"])
    assert torch.allclose(x_train.std(0, correction=0), checkpoint["std"])

    # Settings fixed before any baseline test evaluation; no search or tuning.
    baseline = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=42))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        baseline.fit(data["train"][1], data["train"][2])

    report = {"checkpoint_sha256": digest, "features": FEATURES,
              "versions": dict(python=platform.python_version(), torch=torch.__version__,
                               numpy=np.__version__, sklearn=sklearn.__version__,
                               matplotlib=matplotlib.__version__),
              "baseline": dict(C=1.0, solver="lbfgs", max_iter=1000, random_state=42,
                               iterations=int(baseline[-1].n_iter_[0])),
              "splits": {}, "metrics": {}, "overlap": {}}
    predictions = {}
    for key, (rows, x, y, dates) in data.items():
        report["splits"][key] = dict(n=len(y), occupied=int(y.sum()),
            unoccupied=int((y == 0).sum()), occupied_rate=float(y.mean()),
            start=str(min(dates)), end=str(max(dates)), missing_or_nonfinite=0,
            duplicate_timestamps=len(dates)-len(set(dates)),
            min=dict(zip(FEATURES, x.min(0).tolist())),
            max=dict(zip(FEATURES, x.max(0).tolist())))
        with torch.inference_mode():
            logits = model((torch.tensor(x, dtype=torch.float32)-checkpoint["mean"])/checkpoint["std"])
            probs = torch.sigmoid(logits).numpy().ravel()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor(y[:, None], dtype=torch.float32)).item()
        pred = (probs >= 0.5).astype(int)
        predictions[key] = (probs, pred)
        report["metrics"][key] = {"mlp": dict(metrics(y, pred), loss=loss),
            "logistic_regression": metrics(y, (baseline.predict_proba(x)[:, 1] >= 0.5).astype(int)),
            "always_unoccupied": metrics(y, np.zeros_like(y))}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        shared = len(set(data[a][3]) & set(data[b][3]))
        report["overlap"][a+"_"+b] = shared
        assert shared == 0

    rows, x, y, dates = data["train"]
    report["train_class_medians"] = {str(label): dict(zip(FEATURES, np.median(x[y==label], axis=0).tolist())) for label in [0, 1]}
    q1, q3 = np.quantile(x, [.25, .75], axis=0)
    report["train_iqr_outliers"] = dict(zip(FEATURES, ((x < q1-1.5*(q3-q1)) | (x > q3+1.5*(q3-q1))).sum(0).tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), layout="constrained")
    labels = ["Eğitim", "Doğrulama", "Test"]
    rates = [report["splits"][k]["occupied_rate"]*100 for k in data]
    axes[0].bar(labels, [100-r for r in rates], label="Boş", color="#c7d7e5")
    axes[0].bar(labels, rates, bottom=[100-r for r in rates], label="Dolu", color="#267579")
    for i, rate in enumerate(rates): axes[0].text(i, 100-rate/2, f"%{rate:.1f}", ha="center", va="center", color="white")
    axes[0].set(ylabel="Örneklerin yüzdesi", ylim=(0, 105), title="A  Sınıf dağılımı")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].boxplot([x[y==0, 2], x[y==1, 2]], tick_labels=["Boş", "Dolu"],
                    flierprops=dict(markersize=2, alpha=.25))
    axes[1].set(ylabel="Işık (lux)", title="B  Eğitim verisinde ışık")
    save(fig, "eda_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
    ax.plot(dates, x[:, 3], color="#245b88", linewidth=1, label="CO₂")
    ax.fill_between(dates, 0, 1, where=y==1, transform=ax.get_xaxis_transform(),
                    color="#dfb651", alpha=.3, label="Oda dolu")
    ax.set(ylabel="CO₂ (ppm)", title="C  Eğitim döneminde CO₂ ve doluluk", xlabel="Tarih (2015)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.legend(loc="upper left", ncol=2)
    save(fig, "eda_timeline.png")

    rows, x, y, dates = data["test"]
    probs, pred = predictions["test"]
    cm = np.array(report["metrics"]["test"]["mlp"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.4, 3.2), layout="constrained")
    ax.imshow(cm, cmap="Blues")
    for (i, j), n in np.ndenumerate(cm):
        ax.text(j, i, f"{n:,}".replace(",", "."), ha="center", va="center", fontsize=17,
                color="white" if n>cm.max()/2 else "#182b3b")
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Boş", "Dolu"], yticklabels=["Boş", "Dolu"],
           xlabel="Tahmin edilen sınıf", ylabel="Gerçek sınıf", title="MLP final test confusion matrix")
    save(fig, "confusion_matrix.png")

    groups = {"TN": (y==0)&(pred==0), "FP": (y==0)&(pred==1),
              "FN": (y==1)&(pred==0), "TP": (y==1)&(pred==1)}
    seconds = np.array([d.timestamp() for d in dates])
    changes = np.flatnonzero(y[1:] != y[:-1])+1
    nearest = np.min(np.abs(seconds[:, None]-seconds[changes][None, :]), axis=1)/60
    report["error_groups"] = {}
    for name, mask in groups.items():
        report["error_groups"][name] = dict(n=int(mask.sum()),
            median=dict(zip(FEATURES, np.median(x[mask], axis=0).tolist())),
            near_transition_2min=int((nearest[mask] <= 2).sum()))
    report["transition_definition"] = "Within 120 seconds of the first row with a changed test label; retrospective descriptive analysis only."
    report["error_examples"] = []
    for name in ["FP", "FN"]:
        # First chronological errors, not cherry-picked for confidence or features.
        for i in np.flatnonzero(groups[name])[:3]:
            report["error_examples"].append(dict(type=name, date=str(dates[i]),
                light=float(x[i,2]), co2=float(x[i,3]), probability=float(probs[i]),
                actual=int(y[i]), predicted=int(pred[i])))
    with (OUT / "test_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date"]+FEATURES+["actual", "predicted", "probability", "error_type"])
        for i in np.flatnonzero(y != pred):
            writer.writerow([str(dates[i])]+x[i].tolist()+[int(y[i]), int(pred[i]), float(probs[i]), "FP" if pred[i] else "FN"])

    fig, ax = plt.subplots(figsize=(8.5, 3), layout="constrained")
    for name, color in [("TN", "#b2bec8"), ("TP", "#89bdb6"), ("FP", "#c56b22"), ("FN", "#9c3360")]:
        mask = groups[name]
        ax.scatter(x[mask,2], x[mask,3], s=10 if name in ["TN", "TP"] else 26,
                   alpha=.2 if name in ["TN", "TP"] else .8, color=color,
                   label=f"{name} (n={mask.sum()})", edgecolors="none")
    ax.set(xlabel="Işık (lux)", ylabel="CO₂ (ppm)", title="Test hatalarının sensör uzayındaki dağılımı")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "error_scatter.png")

    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == digest
    assert cm.tolist() == [[7595,108],[13,2036]], "Unexpected supplied-checkpoint results"
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "error_groups": report["error_groups"],
                      "train_class_medians": report["train_class_medians"],
                      "train_iqr_outliers": report["train_iqr_outliers"],
                      "versions": report["versions"], "error_examples": report["error_examples"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

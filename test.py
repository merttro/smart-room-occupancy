import pandas as pd
import torch
import torch.nn as nn
import copy
from model import OccupancyCalc

test_df = pd.read_csv("data/datatest2.txt" ,index_col=0)
test_X = test_df[["Temperature","Humidity","Light","CO2","HumidityRatio"]]
test_y = test_df[["Occupancy"]]

test_X = torch.tensor(test_X.to_numpy(), dtype=torch.float32)
test_y = torch.tensor(test_y.to_numpy(), dtype=torch.float32)

loss_fn = nn.BCEWithLogitsLoss()

checkpoint = torch.load(
    "occupancy_model.pth",
    weights_only=True
)

model = OccupancyCalc()

model.load_state_dict(checkpoint["model_state_dict"])

mean = checkpoint["mean"]
std = checkpoint["std"]

model.eval()
with torch.no_grad():
    test_X_scaled = (test_X - mean) / std



    test_outputs = model(test_X_scaled)
    test_loss = loss_fn(test_outputs,test_y)

    test_probabilities = torch.sigmoid(test_outputs)
    test_predictions = (test_probabilities >= 0.5).float()

    correct = (test_predictions == test_y).sum().item()

    accuracy = (correct / test_y.numel()) * 100

tp = ((test_predictions == 1) & (test_y == 1)).sum().item()
tn = ((test_predictions == 0) & (test_y == 0)).sum().item()
fp = ((test_predictions == 1) & (test_y == 0)).sum().item()
fn = ((test_predictions == 0) & (test_y == 1)).sum().item()
confmat = [[tn,fp],
           [fn,tp]]    
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score =  (precision * recall) / (precision + recall) * 2

print(f"Confusion Matrix: {confmat}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1_score}")

print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: %{accuracy:.2f}")
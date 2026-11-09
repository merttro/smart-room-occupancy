import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import copy
from model import OccupancyCalc

#DataFrame
df = pd.read_csv("data/datatraining.txt" , index_col=0)
val_df = pd.read_csv("data/datatest.txt", index_col=0)

X = df[["Temperature","Humidity","Light","CO2","HumidityRatio"]]

y = df[["Occupancy"]]

val_X = val_df[X.columns]

val_y = val_df[["Occupancy"]]

#Tensor Transition
train_x = torch.tensor(X.to_numpy(), dtype=torch.float32)
val_x = torch.tensor(val_X.to_numpy(), dtype=torch.float32)
train_y = torch.tensor(y.to_numpy(), dtype=torch.float32)
val_y = torch.tensor(val_y.to_numpy(), dtype=torch.float32)

#Scaling
mean = train_x.mean(dim=0)
std = train_x.std(dim=0, correction=0)
std = torch.where(std == 0, 1.0, std)

train_x_scaled = (train_x - mean) / std
val_x_scaled = (val_x - mean) / std

train_dataset = TensorDataset(train_x_scaled,train_y)
validation_dataset = TensorDataset(val_x_scaled,val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


torch.manual_seed(42)
model = OccupancyCalc()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


best_val_loss = float("inf")
best_model = None
patience = 5
counter = 0

for epoch in range(100):
    model.train()
    for x_batch,y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs,y_batch)
        loss.backward()
        optimizer.step()


    model.eval()
    with torch.no_grad():
        val_outputs = model(val_x_scaled)
        val_loss = loss_fn(val_outputs, val_y)
        if best_val_loss >= val_loss.item():
            best_val_loss = val_loss.item()
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                model.load_state_dict(best_model)
                break

    if (epoch + 1) % 10 == 0:

        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
        print(f"W{epoch+1}:  " ,model.layer3.weight)
        print(f"Epoch {epoch+1}: Val Loss = {val_loss.item():.4f}")

        model.train()

#Model Saving
model.load_state_dict(best_model)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "mean": mean,
    "std": std
}

torch.save(checkpoint, "occupancy_model.pth")

print("Model saved")

model.eval()

#Validation
with torch.no_grad():

    train_logits = model(train_x_scaled)
    val_logits = model(val_x_scaled)

    train_loss = loss_fn(train_logits, train_y)
    val_loss = loss_fn(val_logits, val_y)

    val_probabilities = torch.sigmoid(val_logits)

    val_predictions = (val_probabilities >= 0.5).float()

    correct = (val_predictions == val_y).sum().item()

    accuracy = (correct / val_y.numel()) * 100

#Confusion Matrix
    tp = ((val_predictions == 1) & (val_y == 1)).sum().item()
    tn = ((val_predictions == 0) & (val_y == 0)).sum().item()
    fp = ((val_predictions == 1) & (val_y == 0)).sum().item()
    fn = ((val_predictions == 0) & (val_y == 1)).sum().item()
    confmat = [[tn,fp],
               [fn,tp]]    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score =  (precision * recall) / (precision + recall) * 2

print(f"Train Loss: {train_loss.item():.4f}")
print(f"Validation Loss: {val_loss.item():.4f}")
print(f"Validation Accuracy: %{accuracy:.2f}")
print(f"Confusion Matrix: {confmat}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1_score}")

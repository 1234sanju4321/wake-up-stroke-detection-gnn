import pandas as pd
import torch
from model import StrokeNet
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("data/processed/stroke_processed.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = StrokeNet(input_dim=X.shape[1], hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    for xb, yb in loader:
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/stroke_model.pth")
print("Model saved")

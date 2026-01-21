import pandas as pd
import torch
from model import StrokeNet
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/stroke_processed.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

model = StrokeNet(input_dim=X.shape[1], hidden_dim=64)
model.load_state_dict(torch.load("models/stroke_model.pth"))
model.eval()

with torch.no_grad():
    preds = model(torch.Tensor(X)).squeeze().numpy()
    preds = (preds > 0.5).astype(int)

print("Accuracy:", accuracy_score(y, preds))

import numpy as np
import pandas as pd
import torch
from torch import nn

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y_train = train_data['SalePrice'].to_numpy()
train_data = train_data.replace(['NaN','NA'], [0,0])

train_data_modified = train_data[["MSSubClass","LotArea","OverallQual","OverallCond","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea","TotRmsAbvGrd","GarageArea"]]
test_data_modified = test_data[["MSSubClass","LotArea","OverallQual","OverallCond","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea","TotRmsAbvGrd","GarageArea"]]

train_data_modified_numpy = train_data_modified.to_numpy()
test_data_modified_numpy = test_data_modified.to_numpy()
train_data_tensor = torch.from_numpy(train_data_modified_numpy).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
test_data_tensor = torch.from_numpy(test_data_modified_numpy).to(torch.float32)
train_data_tensor = torch.nan_to_num(train_data_tensor)
test_data_tensor = torch.nan_to_num(test_data_tensor)
train_data_tensor_normal = torch.nn.functional.normalize(train_data_tensor, p=2.0, dim=0)
test_data_tensor_normal = torch.nn.functional.normalize(test_data_tensor, p=2.0, dim=0)
y_mean = y_train_tensor.mean()
y_std = y_train_tensor.std()
y_train_tensor_normal = (y_train_tensor - y_mean)/y_std 

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features = 12, out_features = 128)
        self.layer2 = nn.Linear(in_features = 128, out_features = 128)
        self.layer3 = nn.Linear(in_features = 128, out_features = 128)
        self.layer4 = nn.Linear(in_features = 128, out_features = 1)
        self.relu = nn.ReLU()
    def forward(self,X):
        X = self.layer1(X)
        X = self.relu(X)
        X = self.layer2(X)
        X = self.relu(X)
        X = self.layer3(X)
        X = self.relu(X)
        X = self.layer4(X)
        return X
model = Regressor()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15000
for epoch in range(epochs):
    model.train()
    y = model(train_data_tensor_normal).squeeze()
    loss = loss_fn(y, y_train_tensor_normal)
    if torch.isnan(loss).any():
        print(f"NaN detected in loss at epoch {epoch}")
        break
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print("Epoch:", epoch)
    print("Loss", loss)

model.eval()
y_pred = model(test_data_tensor_normal)
y_predictions = (y_pred * y_std) + y_mean
y_predictions.squeeze()

y_preds = y_predictions.detach().numpy().squeeze()
output = pd.DataFrame({"Id": test_data["Id"], "SalePrice": y_preds})
output.to_csv("output.csv", index=False)

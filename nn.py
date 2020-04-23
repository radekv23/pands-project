import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reading the CSV files and passing the filename  to a variable named iris.
iris = pd.read_csv("iris.csv")
print(iris.head())

# Neural Network in PyTorch
# Mapping
mappings = {
   'setosa': 0,
   'versicolor': 1,
   'virginica': 2
}
iris['species'] = iris['species'].apply(lambda x: mappings[x])

# Train & Split
X = iris.drop('species', axis=1).values
y = iris['species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Defining a NN Model
class INN(nn.Module):
   def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(in_features=4, out_features=16)
    self.fc2 = nn.Linear(in_features=16, out_features=12)
    self.output = nn.Linear(in_features=12, out_features=3)
 
   def forward(self, x):
     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))
     x = self.output(x)
     return x

model = INN()
print(model)

# Criterion & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Model Training
epochs = 100
loss_arr = []
for i in range(epochs):
   y_hat = model.forward(X_train)
   loss = criterion(y_hat, y_train)
   loss_arr.append(loss)
 
   if i % 10 == 0:
       print(f'Epoch: {i} Loss: {loss}')
 
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()


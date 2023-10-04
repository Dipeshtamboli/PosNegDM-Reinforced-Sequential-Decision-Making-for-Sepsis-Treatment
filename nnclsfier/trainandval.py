import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from itertools import chain
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("/content/drive/MyDrive/observations.json") as f:
  obs=json.load(f)
X=np.array(obs[0])
y=np.array(obs[1])
y = list(chain(*y))

y=np.where(y==1.0,0,y)
y=np.where(y==-1.0,1,y)

X_test=np.array(obs[2])
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=1)

unique_values, counts = np.unique(y_train, return_counts=True)
class_wts=np.array([1-(counts[0]/(counts[0]+counts[1])),1-(counts[1]/(counts[0]+counts[1]))])
class_wts=torch.tensor(class_wts,dtype=torch.float32).to(device)

bsmote = BorderlineSMOTE(random_state = 1, kind = 'borderline-1')
X_train, y_train = bsmote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert training data to PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

y_train = torch.nn.functional.one_hot(y_train, 2)
y_train = y_train.to(torch.float32).clone().detach().requires_grad_(True)
# Create weighted random sampler based on class weights
weights = 1. / torch.tensor(counts, dtype=torch.float32)
# sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# training_pair = list(zip(X_train, y_train))

# Create DataLoader for training data with weighted random sampling

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

class Mortality(nn.Module):
    def __init__(self, input_dim):
        super(Mortality, self).__init__()
        self.layer1 = nn.Linear(input_dim, 250)
        self.bn1 = nn.BatchNorm1d(250)
        self.dp1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(250, 150)
        self.bn2 = nn.BatchNorm1d(150)
        self.dp2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(150, 75)
        self.bn3 = nn.BatchNorm1d(75)
        self.dp3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(75,25)
        self.bn4 = nn.BatchNorm1d(25)
        self.dp4 = nn.Dropout(0.2)
        self.layer5 = nn.Linear(25,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dp1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dp2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dp3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dp4(x)
        x = self.layer5(x)
        x = self.softmax(x)
        return x

input_dim = X_train.shape[1]
model = Mortality(input_dim)
model.to(device)
num_epochs = 2000
criterion = nn.CrossEntropyLoss(class_wts)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_list=[]
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    print("loss {} curr_epoch {}".format(loss.item(),epoch))
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), '/content/drive/MyDrive/Research/NNMortalityborder.pth')
plt.plot(loss_list)

with torch.no_grad():
    y_pred = torch.argmax(model(X_val.to(device)), dim=1).cpu().numpy()
accuracy = accuracy_score(y_val.numpy(), y_pred)
print("accuracy ",accuracy)

cf=confusion_matrix(y_val.numpy(), y_pred)
# create a figure and axis
fig, ax = plt.subplots()

# plot the confusion matrix as an image
im = ax.imshow(cf, cmap='Blues')


cbar = ax.figure.colorbar(im, ax=ax)


ax.set_xticks(np.arange(cf.shape[1]))
ax.set_yticks(np.arange(cf.shape[0]))
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive'])
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

for i in range(cf.shape[0]):
    for j in range(cf.shape[1]):
        ax.text(j, i, format(cf[i, j], 'd'), ha='center', va='center', color='white' if cf[i, j] > cf.max() / 2 else 'black')

# set title
ax.set_title('Confusion matrix')

# show the plot
plt.show()
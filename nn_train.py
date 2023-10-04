# Importing required packages
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
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
with open("logs/json/observations.json") as f:
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
class_wts=torch.tensor(class_wts,dtype=torch.float32)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train= torch.nn.functional.one_hot(y_train,2)
y_train=torch.tensor(y_train,dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
class Mortality(nn.Module):
    def __init__(self, input_dim):
        super(Mortality, self).__init__()
        self.layer1 = nn.Linear(input_dim, 250)
        self.layer2 = nn.Linear(250, 150)
        self.layer3 = nn.Linear(150, 75)
        self.layer4 = nn.Linear(75,25)
        self.layer5 = nn.Linear(25,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.softmax(x)
        return x

input_dim = X_train.shape[1]
print(f"Input dimension: {input_dim}")
model = Mortality(input_dim)

num_epochs = 500
criterion = nn.CrossEntropyLoss(weight=class_wts)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_record=[]
for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # print(loss)
    loss_record.append(loss)
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0:
        torch.save(model.state_dict(),"nn_model/NNmodel500.pt")

y_pred = torch.argmax(model(X_val), dim=1).numpy()
accuracy = accuracy_score(y_val.numpy(), y_pred)


print("Accuracy:", accuracy)
# torch.save(model,"nn_model/NNmodel500_2.pt")
torch.save(model.state_dict(),"nn_model/NNmodel500.pt")

cf=confusion_matrix(y_val.numpy(), y_pred)
# create a figure and axis
fig, ax = plt.subplots()

# plot the confusion matrix as an image
im = ax.imshow(cf, cmap='Blues')

# add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# set the axis labels
ax.set_xticks(np.arange(cf.shape[1]))
ax.set_yticks(np.arange(cf.shape[0]))
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive'])
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

# loop over data dimensions and create text annotations
for i in range(cf.shape[0]):
    for j in range(cf.shape[1]):
        ax.text(j, i, format(cf[i, j], 'd'), ha='center', va='center', color='white' if cf[i, j] > cf.max() / 2 else 'black')

# set title
ax.set_title('Confusion matrix')

# show the plot
# plt.show()
plt.savefig('confusion_matrix_nn.png')
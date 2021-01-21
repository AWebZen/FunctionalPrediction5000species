#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:14:25 2020

@author: adele
"""
from os.path import join as joinP
import pickle as pk
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
import torch.nn as nn
import pandas as pd

# Device configuration
device = torch.device('cpu')


def train_test_proportions(freq, X, y):
    """
    Split data into training (freq proportion) and testing (1-freq) datasets. Different classes will be represented in
    proportions from y, and observations will be shuffled.

    freq - percentage of data to go to train, between 0 and 1
    X - data
    y - classes
    """
    assert freq < 1 and 0 < freq, "freq must be a proportion"
    c = Counter(y)
    idx = {k:np.where(np.array(y) == k)[0] for k in c.keys()}
    train_idx = []
    test_idx = []
    for k in c.keys():
        n_to_choose_train = int(c[k] * freq + 0.5) # + 0.5 hack to round number to closest int (for positives) like in statistics
        train_idx.extend(list(np.random.choice(idx[k], n_to_choose_train, replace=False)))
        test_idx.extend([x for x in idx[k] if x not in train_idx])
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("Frequence too high or two low, unable to form 2 groups")
        raise SystemExit()
    y_train = np.array(y)[train_idx]
    y_test = np.array(y)[test_idx]
    if len(y) == X.shape[0]:
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
    else:
        X_train = X[:, train_idx]
        X_test = X[:, test_idx]
    return X_train, X_test, y_train, y_test, train_idx, test_idx

def weighted_mse_loss(input, target, crit, weighted=False):
    """
    Create custom Loss of MSE loss (weighted MSE loss) and L1 penality (L1 regulizer).

    weight - of size input, weight of class for each value
    """
    if isinstance(weighted, torch.Tensor):
        return torch.sum(weighted * (input - target) ** 2)
    else :
        return crit(input, target)

# Get data
backup_dir = "backup_cpkl/"
med_name = "PsychroMesoThermHyperMedium"
#Get the scope matrix by running scope_kegg_prk.py
simplified_matrix = pk.load(open(joinP(backup_dir, "simplified_matrix_scope{}_prk.cpk".format(med_name)), "rb"),
                            encoding='latin1') #Scope matrix

df_species = pd.read_csv("species_metadata.csv", index_col=0)
df_species.loc[df_species.sp_codes.isnull(), "sp_codes"] = "nan"  # Otherwise interpreted as NaN

# Standardisation
growth_temp_crT = (df_species.temp_def.dropna().values - np.min(df_species.temp_def.dropna().values))/(np.max(df_species.temp_def.dropna().values) - np.min(df_species.temp_def.dropna().values))
matrix_tempT = simplified_matrix[:, df_species.temp_def.notnull()]
classes_T = df_species.temp_range_deduced[df_species.temp_range_deduced.notnull()].values


# Remove mesophiles: keep only 300
no_meso = sorted(list(np.where(classes_T != "mesophilic")[0]) + list(np.random.choice(np.where(classes_T == "mesophilic")[0], 300, replace=False)))
# no_meso = pk.load(open("backup_cpkl/no_meso_index_list.cpk", "rb")) #Saved index of matrix without temp null values
classes = classes_T[no_meso]
matrix_temp = matrix_tempT[:, no_meso]
growth_temp_cr = growth_temp_crT[no_meso]

# Data
X_train, X_test, _, _, train_idx, test_idx = train_test_proportions(.66, matrix_temp.T, classes) #data, y_classes)
# pk.dump([train_idx, test_idx], open("backup_cpkl/NN_train_test_indices.cpk", "wb"), protocol=2)
# train_idx, test_idx = pk.load(open("backup_cpkl/NN_train_test_indices.cpk", "rb"))
X_train = matrix_temp.T[train_idx, :]
X_test = matrix_temp.T[test_idx, :]
y_tr = growth_temp_cr[train_idx]
y_t = growth_temp_cr[test_idx]

# Divide temps into categories for training dataset to account for weights
n_cats = 10
categories = pd.cut(y_tr, n_cats)
weight_class_dict = {cat : 1/(count/float(len(categories)))
                        for cat, count in Counter(categories).items()
                     }
weights = torch.Tensor([weight_class_dict[cat] for cat in categories])
weights /= torch.sum(weights)


# Hyper-parameters
input_size = len(matrix_temp)
hidden_size = 1000
num_classes = 1
num_epochs = 250
batch_size = 1000
learning_rate = 0.0001


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p=.5):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out



model = NeuralNet(input_size, hidden_size, num_classes, p=.2).to(device)

# Loss and optimizer
criterion = weighted_mse_loss
crit2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
for epoch in range(num_epochs):
    # Move tensors to the configured device
    scope = torch.from_numpy(X_train).float().to(device)
    temp = torch.from_numpy(y_tr).float().to(device)

    model = model.train()
    model.zero_grad()

    # Forward pass
    outputs = model(scope)
    loss = criterion(outputs, temp.unsqueeze(1), crit2, weighted=weights)

    # Backward and optimize
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}],  Loss: {:.4f}'
           .format(epoch+1, num_epochs,  loss.item()/len(y_tr)))

    scope = torch.from_numpy(X_test).float().to(device)
    temp = torch.from_numpy(y_t).float().to(device)

    model = model.eval()
    outputs = model(scope)
    loss = criterion(outputs, temp.unsqueeze(1), crit2, weighted=False)

    print("Test:", loss.item()/len(y_t))


untransform = lambda x:  x * (np.max(df_species.temp_def.dropna().values) - np.min(df_species.temp_def.dropna().values)) + np.min(df_species.temp_def.dropna().values)
plt.figure()
plt.plot(untransform(y_t), untransform(outputs.detach().numpy()), "+")
print(r2_score(untransform(y_t), untransform(outputs.detach().numpy())))
plt.xlabel("Real growth temperature (ºC)")
plt.ylabel("Predicted growth temperature (ºC)")
plt.plot([0,100], [0,100])


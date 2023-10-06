import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def Dat2Ten(dataloader):
    x_total, y_total = [], []
    for x,y in dataloader:
        x_total.append(x)
        y_total.append(y)
    return torch.cat(x_total, dim=0), torch.cat(y_total, dim=0)

def Dat2Num(dataloader):
    x_total, y_total = [], []
    for x,y in dataloader:
        x_total.append(x)
        y_total.append(y)
    return np.concatenate(x_total, axis=0), np.concatenate(y_total, axis=0)

def Num2Dat(X, Y, mini_batch = None):
    data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)) # create your datset
    if mini_batch is None:
        mini_batch = len(X)
    return DataLoader(data, batch_size=mini_batch)

def Num2Ten(x):
        return torch.tensor(np.array(x), dtype=torch.float32)

def Ten2Dat(X, Y, mini_batch = None):
    data = TensorDataset(X,Y) # create your datset
    if mini_batch is None:
        mini_batch = len(X)
    return DataLoader(data, batch_size=mini_batch)

# def Ten2Num(x, detach=True):
#     if detach:
#         return x.detach().numpy()
#     else:
#         return x.numpy()

def Ten2Num(x, detach=True):
    if x.requires_grad:
        return x.detach().numpy()
    else:
        return x.numpy()

def NLLloss(y_real, y_pred, var):
    return (torch.log(var) + ((y_real - y_pred).pow(2))/var).mean()/2 + 0.5*np.log10(2*np.pi)

def read_csv(N_inp, dir="DOEset1.csv"):
    data = pd.read_csv(dir)
    header = list(data.columns)
    inp_dataset = data[header[:N_inp]].values
    out_dataset = data[header[N_inp:]].values

    return header[:N_inp], header[N_inp:], inp_dataset, out_dataset

def csv2Dat(N_inp, dir="DOEset1.csv", mini_batch = None):
    H_inp, H_out, X, Y = read_csv(N_inp, dir)
    return H_inp, H_out, Num2Dat(X, Y, mini_batch)

def csv2Ten(N_inp, dir="DOEset1.csv", mini_batch = None):
    H_inp, H_out, X, Y = read_csv(N_inp, dir)
    return H_inp, H_out, Num2Ten(X), Num2Ten(Y)

def csv2Num(N_inp, dir="DOEset1.csv", mini_batch = None):
    H_inp, H_out, inp_dataset, out_dataset =  read_csv(N_inp, dir)
    return H_inp, H_out, inp_dataset, out_dataset

def normalize(data):
    STD = StandardScaler()
    scaled_data = STD.fit_transform(data)
    return scaled_data, STD

def normalize_multifidelity(data, minmax = True, Scaler=None): # for multi-fidelity data
    scaled_data_list, Scaler_list = [], []
    for x in data: # for loop since "data" is the list of the multi-fidelity data
        if Scaler is not None: # When "Scaler" is given, transform "data" using it
            scaled_data = Scaler.transform(x)
        elif minmax: # When "Scaler" is not given, define new Scaler (MinMaxScaler in this case)
            Scaler = MinMaxScaler()
            scaled_data = Scaler.fit_transform(x)
        else: # When "Scaler" is not given, define new Scaler (StandardScaler in this case)
            Scaler = StandardScaler()
            scaled_data = Scaler.fit_transform(x)
        scaled_data_list.append(scaled_data)
        Scaler_list.append(Scaler)

    return scaled_data_list, Scaler_list, data

def reject_outliers(x, y, k = 3.):
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    distance_from_mean = abs(y - mean)
    not_outlier = distance_from_mean < k * std
    not_outlier = np.all(not_outlier, axis=1)
    return x[not_outlier], y[not_outlier]
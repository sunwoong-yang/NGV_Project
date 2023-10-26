from surrogate_model.GPRs import GPRs
from surrogate_model.DE import DeepEnsemble
import pandas as pd
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
import numpy as np
from PrePost.PrePost import reject_outliers
import torch.optim as optim
import torch.nn as nn

class DDO(): # Data-Driven Optimization
    def __init__(self, proj_name=""):
        self.proj_name = proj_name

    def read_excel(self, file_name = 'LHS180_t1.xlsx', train_ratio=0.8, shuffle_seed=42, coef_outlier=3):
        df_info = pd.read_excel(f'Projects/{self.proj_name}/{file_name}', "info").fillna(np.nan).replace([np.nan], [None])
        df = pd.read_excel(f'Projects/{self.proj_name}/{file_name}', "conv")
        QoI_direction = np.ones(len(df_info["QoI"]))
        self.n_con = 0
        self.value_con = []
        for idx, (min, max) in enumerate(zip(df_info["minimize"], df_info["maximize"])):
            # If QoI is used as objective function
            if (min is None) and (max in ["v", "V", "o", "O"]): # if maximization, direction = -1
                QoI_direction[idx] = -1.
            elif (min in ["v", "V", "o", "O"]) and (max is None): # if minimization, direction = +1
                pass
            elif (min is None) and (max is None): # error: neither max nor min
                raise Exception(f"Max/Min should be determined for {df_info['QoI'][idx]}")
            elif (min in ["v", "V", "o", "O"]) and (max in ["v", "V", "o", "O"]): # error: neither max nor min
                raise Exception(f"Max/Min objective cannot be performed simultaneously for {df_info['QoI'][idx]}")
            # If QoI is used as constraint
            if (min is None) and (isinstance(max, (float, int))): # if constraint is "eaual or larger", direction = -1
                QoI_direction[idx] = -1.
                self.n_con += 1
                self.value_con.append(max)
            elif (isinstance(min, (float, int))) and (max is None): # if constraint is "eaual or less", direction = +1
                self.n_con += 1
                self.value_con.append(min)
            elif (isinstance(min, (float, int))) and (isinstance(max, (float, int))): # error: neither max nor min
                raise Exception(f"Max/Min constraint cannot be performed simultaneously for {df_info['QoI'][idx]}")

        x_num = sum(word.count("DV") for word in list(df.columns)) # Number of design variables (calculated by how many times "DV" appeared in the indices' name)
        self.x_list = ["DV"+f"{i}" for i in range(x_num)]
        self.y_list = list(df_info["QoI"])

        x = df[self.x_list].to_numpy()
        y = df[self.y_list].to_numpy()
        x,y = reject_outliers(x, y, k=coef_outlier)
        self.train_size = int(x.shape[0] * train_ratio)  # 80% for the train data

        self.x, self.y = x, y
        self.n_var = self.x.shape[1]
        self.n_obj = self.y.shape[1] - self.n_con

        self.QoI_direction_obj = QoI_direction[:self.n_obj]
        self.QoI_direction_con = QoI_direction[self.n_obj:]
        self.shuffle(rand_seed=shuffle_seed)

    def shuffle(self, rand_seed=42):

        np.random.seed(rand_seed)
        # Data shuffling
        random_idx = np.arange(self.x.shape[0])
        np.random.shuffle(random_idx)
        x_shuffled, y_shuffled = self.x[random_idx], self.y[random_idx]

        # Train-Test split
        self.x_train, self.y_train = x_shuffled[:self.train_size], y_shuffled[:self.train_size]
        self.x_test, self.y_test = x_shuffled[self.train_size:], y_shuffled[self.train_size:]

        # Manual elimination of outliers
        # self.x_train = np.delete(self.x_train, self.y_train[:,0]>0.01, axis=0)
        # self.y_train = np.delete(self.y_train, self.y_train[:,0]>0.01, axis=0)
        # self.x_test = np.delete(self.x_test, self.y_test[:,0]>0.01, axis=0)
        # self.y_test = np.delete(self.y_test, self.y_test[:,0]>0.01, axis=0)

    def fit(self, model="GPR", **kwargs):
        self.model_type = model
        if model == "GPR":
            kernel = ConstantKernel() * Matern(length_scale=[1.]*self.n_var, nu=2.5)
            # kernel = ConstantKernel() * RBF(length_scale=[1.]*self.n_var)
            # kernel = ConstantKernel() * RBF(length_scale=1.)
            self.model = GPRs(**kwargs)
            self.model.fit(self.x_train, self.y_train, )
        elif model == "DE":
            layers = kwargs["layers"]
            lr = kwargs["lr"]
            iter = kwargs["iter"]
            # {"optimizer": "adam", "lr": [1e-3] * 3, "loss_weights": [1] * 7}
            self.model = DeepEnsemble(self.n_var, layers, "GELU", self.n_obj+self.n_con, num_models=5)
            criterion_ = nn.MSELoss()

            if len(lr) != len(iter):
                raise Exception("Length of lists 'lr' and 'iter' do not match")
            else:
                for lr_, iter_ in zip(lr, iter):
                    optimizer_ = optim.Adam(self.model.parameters(), lr=lr_)
                    self.model.fit(self.x_train, self.y_train, iter_, optimizer_)
    def predict(self, x=None, return_std=False):
        # if self.model_type == "GPR":
        if x is None: # When the x for the prediction is not given, perform prediction with x_test datatset
            y_pred = self.model.predict(self.x_test, return_std=return_std)
        else:
            y_pred = self.model.predict(x, return_std=return_std)
        # elif self.model_type == "DE":
        #     if x is None: # When the x for the prediction is not given, perform prediction with x_test datatset
        #         y_pred = self.model.predict(self.x_test, return_std=return_std)
        #     else:
        #         y_pred = self.model.predict(x, return_std=return_std)

        return y_pred


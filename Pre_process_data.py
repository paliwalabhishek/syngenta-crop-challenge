import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from util import Constant


class preProcessData:

    @staticmethod
    def preProcessTrainData(self, file):
        raw_trainDF = pd.read_csv(file)
        raw_trainDF = raw_trainDF.drop(columns=[Constant.YEAR, Constant.LOCATION])
        raw_trainDF = raw_trainDF.replace({Constant.INBRED: Constant.Inbred_}, {Constant.INBRED: ''}, regex=True)
        raw_trainDF = raw_trainDF.replace({Constant.INBRED_CLUSTER: Constant.Cluster}, {Constant.INBRED_CLUSTER: ''}, regex=True)
        raw_trainDF = raw_trainDF.replace({Constant.TESTER: Constant.Tester_}, {Constant.TESTER: ''}, regex=True)
        raw_trainDF = raw_trainDF.replace({Constant.TESTER_CLUSTER: Constant.Cluster}, {Constant.TESTER_CLUSTER: ''}, regex=True)
        raw_trainDF = raw_trainDF.groupby([Constant.INBRED, Constant.INBRED_CLUSTER, Constant.TESTER, Constant.TESTER_CLUSTER]).mean().reset_index()

        raw_trainDF = raw_trainDF.astype({Constant.INBRED: Constant.float32, Constant.INBRED_CLUSTER: Constant.float32, Constant.TESTER: Constant.float32,
                                          Constant.TESTER_CLUSTER: Constant.float32})

        x_dataset = np.asarray(raw_trainDF.drop(columns=[Constant.YIELD]))
        y_dataset = np.asarray(raw_trainDF[Constant.YIELD])
        X_train, X_val, Y_train, Y_val = train_test_split(x_dataset, y_dataset, test_size=0.05)
        return X_train, Y_train, X_val, Y_val

    @staticmethod
    def convert_to_tensor(X, Y):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_y = torch.from_numpy(Y)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from NN_BN import Network as NN_bn
from NN_Dropout import Network as NN_dropout
from NN_NBN import Network as NN_no_bn

class Train_Manager:
    def __init__(self):
        self.model = None

    def train_data_set(self, data_set, run, model_path, type_of_model, device):
        model = self.__getModel(type_of_model, device)
        model_updated = self.__load_model(model, data_set, run, model_path, type_of_model, device)
        return model_updated

    def __getModel(self, type_of_model, device):
        if type_of_model == "BatchNorm":
            print("Training with batch Normalization")
            return NN_bn().to(device=device)
        elif type_of_model == "NoBatchNorm":
            print("Training without batch Normalization")
            return NN_no_bn().to(device=device)
        elif type_of_model == "Dropout":
            print("Training with Dropout")
            return NN_dropout().to(device=device)

    def __load_model(self, model, data_set, run, model_path, type_of_model, device):
        if os.path.isfile(model_path):
            # load trained model parameters from disk
            model.load_state_dict(torch.load(model_path, map_location=device))
            print('Loaded model parameters from disk.')
        else:
            model = self.__train_network(model, data_set, run, type_of_model, device, model_path)
            print('Finished Training.')
            torch.save(model.state_dict(), model_path)
            print('Saved model parameters to disk.')

        return model

    def __train_network(self, model, data_set, run, type_of_model, device, model_path, epochs):
        batch_size = run.batch_size
        lr = run.lr
        shuffle = run.shuffle
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=1,
                                                  pin_memory=True)
        save_file_name = model_path
        tb_summary = self.__get_tb_summary_title(type_of_model)

        # set optimizer - Adam
        optimizer = optim.Adam(model.parameters(), lr=lr)

        torch.backends.cudnn.enabled = False
        for epoch in range(epochs):
            for batch in data_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                # forward propagation
                predictions = model(x)

                loss = F.cross_entropy(predictions, y)

                # zero out grads for every new iteration
                optimizer.zero_grad()

                # back propagation
                loss.backward()

                # update weights
                # w = w - lr * grad_dw
                optimizer.step()

        return model



    @staticmethod
    def __get_tb_summary_title(type_of_model):
        if type_of_model == "BatchNorm":
            return "With-Batch_Normalization-"
        elif type_of_model == "NoBatchNorm":
            return "Without-Batch_Normalization-"
        elif type_of_model == "Dropout":
            return "WithDropout_BN-"


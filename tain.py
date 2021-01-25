import torch
from util import Constant
from Pre_process_data import preProcessData
from trainManager import Train_Manager

def test_with_diff_params():
    run = {}
    x_train, y_train, x_val, y_val = preProcessData.preProcessTrainData(Constant.TRAIN_FILE_PATH)
    train_set = preProcessData.convert_to_tensor(x_train, y_train)
    val_set = preProcessData.convert_to_tensor(x_val, y_val)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = Train_Manager()
    type_of_model = Constant.BatchNorm
    train.train_data_set(train_set, run, Constant.MODEL_PATH , type_of_model, device)



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    test_with_diff_params()

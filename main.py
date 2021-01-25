import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import os

train_file = 'CC2020_data/CC2020_train_final.csv'
test_file = 'CC2020_data/CC2020_test_final.csv'
model_path = 'model/nnModel.pt'


def preProcessTrainData(file):
    raw_trainDF = pd.read_csv(file)
    raw_trainDF = raw_trainDF.drop(columns=['YEAR', 'LOCATION'])
    raw_trainDF = raw_trainDF.replace({'INBRED': 'Inbred_'}, {'INBRED': ''}, regex=True)
    raw_trainDF = raw_trainDF.replace({'INBRED_CLUSTER': 'Cluster'}, {'INBRED_CLUSTER': ''}, regex=True)
    raw_trainDF = raw_trainDF.replace({'TESTER': 'Tester_'}, {'TESTER': ''}, regex=True)
    raw_trainDF = raw_trainDF.replace({'TESTER_CLUSTER': 'Cluster'}, {'TESTER_CLUSTER': ''}, regex=True)
    raw_trainDF = raw_trainDF.groupby(['INBRED', 'INBRED_CLUSTER', 'TESTER', 'TESTER_CLUSTER']).mean().reset_index()

    raw_trainDF = raw_trainDF.astype({'INBRED': 'float32', 'INBRED_CLUSTER': 'float32', 'TESTER': 'float32',
                                      'TESTER_CLUSTER': 'float32'})

    x_dataset = np.asarray(raw_trainDF.drop(columns=['YIELD']))
    y_dataset = np.asarray(raw_trainDF['YIELD'])
    X_train, X_val, Y_train, Y_val = train_test_split(x_dataset, y_dataset, test_size=0.05)
    '''
    x_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(Y_train)
    x_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(Y_val)
    y_train = y_train.type(torch.float32)
    y_train = y_train.view(y_train.size()[0], 1)
    y_val = y_val.type(torch.float32)
    y_val = y_val.view(y_val.size()[0], 1)
    '''
    '''
    x = torch.from_numpy(np.asarray(raw_trainDF.drop(columns=['YIELD'])))
    y = torch.from_numpy(np.asarray(raw_trainDF['YIELD']))
    '''
    return X_train, Y_train, X_val, Y_val


class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 10)
        self.act1 = nn.ReLU()  # Activation function
        self.linear2 = nn.Linear(10, 1)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = torch.nn.Linear(4, 10)  # 13-(10-10)-1
    self.hid2 = torch.nn.Linear(10, 10)
    self.oupt = torch.nn.Linear(10, 1)
    torch.nn.init.xavier_uniform_(self.hid1.weight)  # glorot
    torch.nn.init.zeros_(self.hid1.bias)
    torch.nn.init.xavier_uniform_(self.hid2.weight)
    torch.nn.init.zeros_(self.hid2.bias)
    torch.nn.init.xavier_uniform_(self.oupt.weight)
    torch.nn.init.zeros_(self.oupt.bias)
  def forward(self, x):
    z = torch.tanh(self.hid1(x)).cuda()
    z = torch.tanh(self.hid2(z)).cuda()
    z = self.oupt(z).cuda()  # no activation, aka Identity()
    return z

def accuracy(model, data_x, data_y, pct_close):
  n_items = len(data_y)
  X = torch.Tensor(data_x).cuda()  # 2-d Tensor
  Y = torch.Tensor(data_y).cuda()  # actual as 1-d Tensor
  oupt = model(X)       # all predicted as 2-d Tensor
  pred = oupt.view(n_items)  # all predicted as 1-d
  n_correct = torch.sum((torch.abs(pred - Y) < torch.abs(pct_close * Y)))
  result = (n_correct.item() * 100.0 / n_items)  # scalar
  return result, pred


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out


x_train, y_train, x_val, y_val = preProcessTrainData(train_file)
'''
train_ds = TensorDataset(x_train, y_train)
batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
loss_fn = torch.nn.MSELoss()
'''
'''Neural Net'''
'''
model = SimpleNet()
opt = torch.optim.SGD(model.parameters(), 1e-5)
print('----------------Training Neural Network------------------------')
for epoch in range(10):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        #print('epoch {}, loss {}'.format(epoch, loss.data))
        

print('----------------Result from Neural Network------------------------')
preds = model(x_val)
for i in range(len(preds)):
    print('Pred: '+ str(preds[i].data) + '  Actual:' + str(y_val[i]))
'''
'''Testing 2nd Net Model'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
net = Net()
net.cuda(device)
net = net.train()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), 0.01)
bat_size = 10
n_items = len(x_train)
batches_per_epoch = n_items
max_batches = 100 * batches_per_epoch
if not os.path.isfile(model_path):
    print('----------------Training 2nd Neural Network------------------------')
    for b in range(max_batches):
        curr_bat = np.random.choice(n_items, bat_size, replace=False)
        X = torch.Tensor(x_train[curr_bat]).cuda(device)
        Y = torch.Tensor(y_train[curr_bat]).view(bat_size, 1).cuda(device)
        optimizer.zero_grad()
        oupt = net(X)
        loss_obj = loss_func(oupt, Y)
        loss_obj.backward()
        optimizer.step()
        if b % (max_batches // 10) == 0:
            print("batch = %6d" % b, end="")
            print("  batch loss = %7.4f" % loss_obj.item(), end="")
            net = net.eval()
            acc = accuracy(net, x_train, y_train, 0.15)
            net = net.train()
            print("  accuracy = %0.2f%%" % acc)

    print("Training complete \n")
    print("Saving Model")
    torch.save(net, model_path)
else:
    net = torch.load(model_path)

print('----------------Result from 2nd  Neural Network------------------------')
X_Test = torch.Tensor(x_val).cuda(device)
Y_Test = torch.Tensor(y_val).cuda(device)
net = net.eval()
acc, pred = accuracy(net, x_val, y_val, 0.10)
print("Overall Accuracy := %0.2f%%" % acc)
print(pred)
'''
preds = net(X_Test)
for i in range(len(preds)):
    print('Pred: '+ str(preds[i].item()) + '  Actual:' + str(y_val[i]))
'''

'''Simple Regrs'''
'''
model_lin = LinearRegressionModel(4, 1)
criterion = torch.nn.MSELoss()
opt_lin = torch.optim.SGD(model_lin.parameters(), 1e-5)

print('----------------Training Regression Model------------------------')
for epoch in range(10):
    for xb, yb in train_dl:
        input = Variable(xb)
        ouput = Variable(yb)
        opt_lin.zero_grad()
        pred = model_lin.forward(input)
        print(pred)
        loss = criterion(pred, ouput)
        loss.backward()
        opt_lin.step()
        print('epoch {}, loss {}'.format(epoch, loss.data))
'''

'''

print('----------------Result from Regression Model------------------------')
pred = model_lin.forward(x_val)
for i in range(len(pred)):
    print('Pred: '+ str(pred[i].data) + '  Actual:' + str(y_val[i]))
'''

'''
parameters = {'x': x_train, 'y': y_train, 'w': w, 'b': b}
preds = model(parameters)
loss = mse(preds, parameters['y'])
loss.backward()
print(loss)
print(parameters['w'])
print(parameters['w'].grad)
'''
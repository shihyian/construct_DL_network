import torch
import torch.nn as nn
import scipy.io

# 設定用GPU/CPU跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('current device: ' + str(device))

# 全連接神經網路，最後output dimension只有1，所以用logistic regression的方式
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.iLayer = nn.Linear(2, 3)
        self.hLayer1 = nn.Linear(3, 2)
        self.hLayer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.iLayer(x))
        x = torch.relu(self.hLayer1(x))
        x = torch.sigmoid(self.hLayer2(x))
        return x

model = NN() # 建立model
model.to(device) # model跟資料丟到gpu跑
print(model)

# load data
# training data
train_mat = scipy.io.loadmat('train.mat') # numpy.ndarray
train_x1 = torch.from_numpy(train_mat['x1']).float()
train_x2 = torch.from_numpy(train_mat['x2']).float()
train_y = torch.from_numpy(train_mat['y']).float()
train_data = torch.cat((train_x1, train_x2), 1)
# print(train_x1)

# testing data
test_mat = scipy.io.loadmat('test.mat') # numpy.ndarray
test_x1 = torch.from_numpy(test_mat['x1']).float()
test_x2 = torch.from_numpy(test_mat['x2']).float()
test_y = torch.from_numpy(test_mat['y']).float()
test_data = torch.cat((test_x1, test_x2), 1)
# print(test_y.shape)

# 變數設定
learning_rate = 0.01

# Loss function
Loss = nn.BCELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the network
for epoch in range(1000):
    train_data = train_data.to(device)
    train_y = train_y.to(device)

    # forward pass
    y_pred = model(train_data)
    # print(y_pred.shape)
    cost = Loss(y_pred, train_y)
    

    # backpropagation
    optimizer.zero_grad()
    cost.backward()
    if((epoch + 1) % 100 == 0):
        print('epoch ' + str(epoch + 1) + ': loss is ' + str(cost.item()))

    # gradient descent
    optimizer.step()

# test
with torch.no_grad():
    test_data = test_data.to(device)
    test_y = test_y.to(device)
    output = model(test_data)
    y_pred = torch.round(output)

    # test error
    test_err = ((test_y.shape[0] - torch.sum(y_pred == test_y)) / test_y.shape[0]) * 100
    print(f'test error: {test_err:.3f}%')


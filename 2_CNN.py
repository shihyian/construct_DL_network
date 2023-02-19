import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import random

# 設定用GPU/CPU跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('current device: ' + str(device))

# load mnist data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# normalization
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# X_train切出0.2比例的validation data
features_train, features_test, targets_train, targets_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

# 切好的data轉成tensor
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

# 打包成dataset
val_train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
val_test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# 一些參數設定
learning_rate = 0.01
batch_size = 64
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)
print('run for ' + str(num_epochs) + ' epoches')

# pytorch DataLoader
val_train_loader = torch.utils.data.DataLoader(val_train, batch_size = batch_size, shuffle = True)
val_test_loader = torch.utils.data.DataLoader(val_test, batch_size = batch_size, shuffle = True)

# 最後拿來testing的data
test_data = torch.from_numpy(X_test)
test_y = torch.from_numpy(Y_test).type(torch.LongTensor)
test = torch.utils.data.TensorDataset(test_data,test_y)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

# 有雜訊的testing data
def create_noise_test(X_test, corrupt_lv):
    img_size = 28*28
    noise_data = []
    for i in range(len(X_test)):
        ran_seq = random.sample([n for n in range(img_size)], np.int64(img_size*corrupt_lv))
        x = X_test[i].reshape(-1, img_size)
        x[0, ran_seq]=255
        new_x = x.reshape(28, 28)
        noise_data.append(new_x)
    
    # 打包並轉換成DataLoader
    noise_data = np.array(noise_data)
    print(noise_data.shape)
    return noise_data
    # print(X_test.shape)
    # print(noise_X_test.shape)

# 5 layer cnn (conv1 -> pool1 -> conv2 -> pool2 -> 全連接層)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 5層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0) # output_shape=(16,24,24)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # output_shape=(16,12,12)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # output_shape=(32,8,8)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # output_shape=(32,4,4)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = CNN()
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
input_shape = (-1,1,28,28)

# training
def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader):
    # Traning the Model
    #history-like list for store loss & acc value
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            # 用GPU跑
            images = images.to(device)
            labels = labels.to(device)
            # 1.Define variables
            train = Variable(images.view(input_shape))
            labels = Variable(labels)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(train)
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            correct_train += (predicted == labels).float().sum()
        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for images, labels in test_loader:
            # 用GPU跑
            images = images.to(device)
            labels = labels.to(device)
            # 1.Define variables
            test = Variable(images.view(input_shape))
            # 2.Forward propagation
            outputs = model(test)
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels)
            # 6.Total correct predictions
            correct_test += (predicted == labels).float().sum()
        #6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    return training_loss, training_accuracy, validation_loss, validation_accuracy

# testing
def testing(model, input_shape, test_loader):
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 用GPU跑
            images = images.to(device)
            labels = labels.to(device)
            # 1.Define variables
            test = Variable(images.view(input_shape))
            # 2.Forward propagation
            outputs = model(test)
            # 3.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 4.Total number of labels
            total_test += len(labels)
            # 5.Total correct predictions
            correct_test += (predicted == labels).float().sum()
        
        test_accuracy = 100 * correct_test / float(total_test)
        print(f'test accuracy: {test_accuracy:.3f}%')

# train model with normal data
# print('\n----------start training(normal data)----------\n')
# training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, val_train_loader, val_test_loader)
# print('\n----------end training----------\n')

# train model with noise data
# new_X_train =  create_noise_test(X_train, 0.05)
# features_train, features_test, targets_train, targets_test = train_test_split(new_X_train, Y_train, test_size = 0.2, random_state = 42)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

# 打包成dataset
val_train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
val_test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

val_train_loader = torch.utils.data.DataLoader(val_train, batch_size = batch_size, shuffle = True)
val_test_loader = torch.utils.data.DataLoader(val_test, batch_size = batch_size, shuffle = True)

print('\n----------start training(noise data)----------\n')
training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, val_train_loader, val_test_loader)
print('\n----------end training----------\n')

# test model with normal data
# print('\n----------start testing(normal data)----------')
# testing(model, input_shape, test_loader)

# test model with noise data
print('\n----------start testing(noise data)----------')
# new_X_test = create_noise_test(X_test, 0.05) # 5% noise
# test_data = torch.from_numpy(new_X_test)
test_y = torch.from_numpy(Y_test).type(torch.LongTensor)
test = torch.utils.data.TensorDataset(test_data,test_y)
noise_test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)
testing(model, input_shape, noise_test_loader)
import os
import struct
import numpy as np
from scipy.special import expit
import random
import torch
import torch.nn as nn
import torch.nn.functional as F



def load_mnist(path, kind='train'):
  #"""Load MNIST data from `path`"""
  labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
  images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
  with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II',lbpath.read(8))
    labels = np.fromfile(lbpath,dtype=np.uint8)

    label = np.zeros((len(labels),10),dtype=np.float32)
    for i in range(len(labels)):
    	label[i][labels[i]] = 1


  with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
    images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    image = np.array(images,dtype=np.float32)

  return image, label


def create_dataset():
	trainX,trainY = load_mnist('./data/','train')
	testX,testY=load_mnist('./data/','test')
	print("File read complete.")
	trainXX = np.zeros((60000,1024),dtype=np.float32)
	testXX = np.zeros((10000,1024),dtype=np.float32)
	for i in range(60000):
		for j in range(784):
			trainXX[i][j] = trainX[i][j]
	for i in range(10000):
		for j in range(784):
			testXX[i][j] = testX[i][j]
	print("Size change  complete")
	return trainXX,trainY,testXX,testY

def maxindex(y):
	#y = y.numpy()
	index = 0
	maxx = 0
	for i in range(10):
		if y[0][i].item()>maxx:
			maxx = y[0][i].item()
			index = i
	return index

#trainX,trainY,testX,testY = create_dataset()
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 展开成一维
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #整体过程：卷积，池化，卷积，池化，汇聚汇聚汇聚
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train():
	trainX,trainY,testX,testY=create_dataset()
	net = Net()
	learning_rate = 0.01
	#整个训练集走一遍
	for i in range(60000):
		x = torch.from_numpy(trainX[i].reshape(1,1,32,32))
		y = net(x)
		y_ = torch.from_numpy(trainY[i].reshape(1,10))
		criterion = nn.MSELoss()
		loss = criterion(y, y_)
		#loss_num = loss.item()
		#print(loss)
		net.zero_grad()
		loss.backward()
		for f in net.parameters():
			f.data.sub_(f.grad.data * learning_rate)


		if i % 20 == 0:
			losss = 0
			acc = 0
			for j in range(1000):
				k = random.randint(0,9999)
				y=net(torch.from_numpy(testX[k].reshape(1,1,32,32)))
				y_=torch.from_numpy((testY[k].reshape(1,10)))
				criterion = nn.MSELoss()
				loss = criterion(y,y_)
				losss = losss+loss.item()/1000

				if maxindex(y) == maxindex(y_):
					acc = acc+0.001
			print("%d samples trained, loss is %f, accuracy on test is %f" % (i,losss,acc))
	torch.save(net, 'net.pkl')

	# 重新加载模型
	#net = torch.load('net.pkl')

train()









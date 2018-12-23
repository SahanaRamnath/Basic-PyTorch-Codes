'''
----------------------------------------------
CLASSIFICATION OF CIFAR-10 DATASET USING A CNN
----------------------------------------------
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Dataset
# loading and normalizing CIFAR-10 using module torchvision
transform=transforms.Compose([transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# get train data
train_set=torchvision.datasets.CIFAR10(root='./data',train=True,
	download=True,transform=transform)
# loading to iterate through the data
train_loader=torch.utils.data.DataLoader(train_set,batch_size=4,
	shuffle=True,num_workers=2)

# similarly for test
test_set=torchvision.datasets.CIFAR10(root='./data',train=False,
	download=True,transform=transform)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=4,
	shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# Defining the CNN
class Net(nn.Module) : # inherit from class nn.Module

	def __init__(self) : 
		super(Net,self).__init__()

		# Convolutional Layers
		self.conv1=nn.Conv2d(3,6,5)
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5)
		# Linear Layers
		self.fc1=nn.Linear(in_features=16*5*5,out_features=120,bias=True)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)

	def forward(self,x) : # call this to do a forward pass

		# expected input is image of size 32 x 32

		# Max pooling over a 2 x 2 window
		x=self.pool(F.relu(self.conv1(x)))
		# if input is a square, can give only a single number for window size
		x=self.pool(F.relu(self.conv2(x)))
		# reshape before passing to feed forward network
		x=x.view(-1,16*5*5)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x


net=Net()

# Loss Function and Optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

# Train Loop
for epoch in range(2) : 
	running_loss=0.0
	for i,data in enumerate(train_loader,0) : 

		# inputs
		inputs,labels=data

		optimizer.zero_grad()

		outputs=net(inputs)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		# printing stats
		running_loss+=loss.item()
		if i%2000==1999 : # prints every 2000 batches
			print 'Epoch : ',epoch+1,' Step : ',i+1,' Loss : ',running_loss/2000
			running_loss=0.0
print 'Finished training'

# Testing the network
correct=0
total=0
with torch.no_grad() : 
	for data in test_loader : 
		images,labels=data
		outputs=net(images)
		_,predicted=torch.max(outputs.data,1)
		total+=labels.size(0)
		correct+=(predicted==labels).sum().item()

print 'Accuracy of trained network on 10000 images : ',100*correct/total

# Accuracy class-wise
class_correct=list(0. for i in range(10))
class_total=list(0. for i in range(10))

with torch.no_grad() : 
	for data in test_loader : 
		images,labels=data
		outputs=net(images)
		_,predicted=torch.max(outputs.data,1)
		c=(predicted==labels).squeeze()
		for i in range(4) : # batch size is 4
			label=labels[i]
			class_correct[label]+=c[i].item()
			class_total[label]+=1
print '\nClasswise accuracy : '
for i in range(10) : 
	print 'Accuracy of ',classes[i],' : ',100*class_correct[i]/class_total[i]


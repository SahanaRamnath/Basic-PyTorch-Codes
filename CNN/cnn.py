'''
----------------------------
NEURAL NETWORKS WITH PYTORCH
----------------------------
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# neural nets can be constructed using torch.nn

class Net(nn.Module) : # inherit from class nn.Module

	def __init__(self) : 
		super(Net,self).__init__()

		# Convolutional Layers
		# 1 input image channel
		# 6 output channels (number of filters used)
		# 5 x 5 square convolution kernel
		self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,
			stride=1,padding=0,dilation=1,bias=True)
		self.conv2=nn.Conv2d(6,16,5)
		# creates weight of size [out_channels,in_channels,kernel_size[0],kernel_size[1]]
		# and bias of size [out_channels]

		# Linear Layers
		self.fc1=nn.Linear(in_features=16*5*5,out_features=120,bias=True)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)

	def forward(self,x) : # call this to do a forward pass

		# expected input is image of size 32 x 32

		# Max pooling over a 2 x 2 window
		x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		# if input is a square, can give only a single number for window size
		x=F.max_pool2d(F.relu(self.conv2(x)),2)
		# reshape before passing to feed forward network
		x=x.view(-1,self.num_flat_features(x))
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x

	def num_flat_features(self,x) : 
		size=x.size()[1:] # all dimensions except the batch dimension
		num_features=1
		for s in size : 
			num_features*=s;
		return num_features


net=Net()
print net
# the 'backward' function for backprop is automatically defined in autograd

# Parameters
params=list(net.parameters())
print 'Number of parameters : ',len(params)
print 'Size of first parameter : ',params[0].size()

# Forward pass
input=torch.randn(1,1,32,32)
out=net(input)
print out

# Make the gradient buffers of all parameters zero 
# and fill backward with random gradients
net.zero_grad()
out.backward(torch.randn(1,len(params)))

# Loss Function
target=torch.randn(10) # dummy target
target=target.view(1,-1) # same shape as out
criterion=nn.MSELoss() # mean squared error
out=net(input)
loss=criterion(out,target)
print 'Loss : ',loss.item(),'[',loss,']'

# a few steps backward in the graph from loss
print '\nGoing backwards from the loss : '
print loss.grad_fn # the MSE loss
print loss.grad_fn.next_functions[0][0] # linear
print loss.grad_fn.next_functions[0][0].next_functions[0][0] # relu

# Backpropogation
net.zero_grad()
print 'Gradient of conv1\'s before backward : ',net.conv1.bias.grad
loss.backward()
print 'Gradient of conv1\'s after backward : ',net.conv1.bias.grad

# Updating the weights

# manual way for normal gradient descent
# learning_rate=0.001
# for p in net.parameters() : 
# 	f.data.sub_(f.grad.data*learning_rate)

# using optimizers
optimizer=optim.SGD(net.parameters(),lr=0.001) # or Nesterov-SGD,Adam,RMSProp
# in the train loop
optimizer.zero_grad() # as before
output=net(input)
loss=criterion(output,target)
#loss.backward()
optimizer.step() # performs the update
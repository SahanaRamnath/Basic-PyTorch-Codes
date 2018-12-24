'''
-------------------------------------------
CLASSIFYING NAMES USING CHARACTER LEVEL RNN
-------------------------------------------
'''

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

import unicodedata
import string
import random
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#########################################################################################
# Preprocessing
def find_files(path): 
	return glob.glob(path)
# print(find_files('data/names/*.txt'))

all_letters=string.ascii_letters+" .,;'"
num_letters=len(all_letters)
def unicode_to_ascii(s) : 
	return ''.join(c for c in unicodedata.normalize('NFD',s) 
		if unicodedata.category(c)!='Mn' and c in all_letters)

# build dictionary of category(language) and lines(names)
category_lines={}
all_categories=[]

def read_lines(filename) : 
	lines=open(filename,encoding='utf-8').read().strip().split('\n')
	return [unicode_to_ascii(line) for line in lines]

for file_name in find_files('data/names/*.txt') : 
	category=os.path.splitext(os.path.basename(file_name))[0]
	all_categories.append(category)
	lines=read_lines(file_name)
	category_lines[category]=lines
num_categories=len(all_categories)

# turn names into tensors
# using one hot vectors
# turning into dataset of size : line_length x batch_size x num_letters
def index_of_letter(letter) : # returns index of letter WRT list of all letters
	return all_letters.find(letter)
def letter_to_tensor(letter) : # converts one letter into its one hot vector
	tensor=torch.zeros(1,num_letters)
	tensor[0][index_of_letter(letter)]=1
	return tensor
def line_to_tensor(line) : # converts one line into a matrix of one hot vectors
	tensor=torch.zeros(len(line),1,num_letters)
	for l,letter in enumerate(line) : 
		tensor[l][0][index_of_letter(letter)]=1
	return tensor

#########################################################################################
# Defining the RNN
class RNN(nn.Module) : 

	def __init__(self,input_size,hidden_size,output_size) : 
		super(RNN,self).__init__()

		self.hidden_size=hidden_size
		self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
		self.i2o=nn.Linear(input_size+hidden_size,output_size)
		self.softmax=nn.LogSoftmax(dim=1)

	def forward(self,input,hidden) : 
		combined=torch.cat((input,hidden),1)
		hidden=self.i2h(combined)
		output=self.i2o(combined)
		output=self.softmax(output)
		return output,hidden
	def init_hidden(self) : 
		return torch.zeros(1,self.hidden_size)

num_hidden_units=128
rnn=RNN(num_letters,num_hidden_units,num_categories)

#########################################################################################
# Interpretability for training

def category_from_output(output) : 
	top_n,top_i=output.topk(1) # returns values, indices
	category_i=top_i[0].item()
	return all_categories[category_i],category_i
def random_choice(l) : 
	return l[random.randint(0,len(l)-1)]
def random_train_point() : 
	category=random_choice(all_categories)
	line=random_choice(category_lines[category])
	category_tensor=torch.tensor([all_categories.index(category)],dtype=torch.long)
	line_tensor=line_to_tensor(line)
	return category,line,category_tensor,line_tensor
print('\n10 random train points : ')
print('-------------------------')
for i in range(10) : 
	category,line,category_tensor,line_tensor=random_train_point()
	print('Category : ',category,' Line : ',line)

#########################################################################################
# Training the network using negative log likelihood loss
learning_rate=0.005
criterion=nn.NLLLoss()
def train(category_tensor,line_tensor) : 
	hidden=rnn.init_hidden()
	rnn.zero_grad()
	for i in range(line_tensor.size()[0]) : # pass for every character
		output,hidden=rnn(line_tensor[i],hidden)
	loss=criterion(output,category_tensor)
	loss.backward()

	# update weights based on gradients
	for p in rnn.parameters() : 
		p.data.sub_(learning_rate*p.grad.data)

	return output,loss.item()

num_train_steps=100000
steps_per_stats=5000
steps_per_plot=1000

# for plotting
current_loss=0
all_losses=[]

start=time.time()
print('\n')
for i in range(num_train_steps) : 
	category,line,category_tensor,line_tensor=random_train_point()
	output,loss=train(category_tensor,line_tensor)
	current_loss+=loss

	if i%steps_per_stats==0 : 
		guess,guess_i=category_from_output(output)
		flag='Wrong'
		if guess==category : 
			flag='Correct'
		print('Global Step : ',i,' ',i/float(num_train_steps)*100,'% training over')
		print('Loss : ',loss)
		print('Name : ',line,', Category guessed : ',guess,',\t'+flag+' guess')
	if i>0 and i%steps_per_plot==0 : 
		all_losses.append(current_loss/steps_per_plot)
		current_loss=0

#########################################################################################
# Plotting the loss
plt.figure()
plt.title('Loss vs Train Steps')
plt.plot(all_losses)
plt.xlabel('Train Steps')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.show()

#########################################################################################
# Examining the performance in each language using a confusion matrix
confusion=torch.zeros(num_categories,num_categories)
num_confusion=10000 # evaluating the confusion matrix based on 10000 random data points

def evaluate(line_tensor) : 
	hidden=rnn.init_hidden()
	for i in range(line_tensor.size()[0]) : 
		output,hidden=rnn(line_tensor[i],hidden)
	return output
for i in range(num_confusion) : 
	category,line,category_tensor,line_tensor=random_train_point()
	output=evaluate(line_tensor)
	guess,guess_i=category_from_output(output)
	category_i=all_categories.index(category)
	confusion[category_i][guess_i]+=1
# normalizing each row
for i in range(num_categories) : 
	confusion[i]=confusion[i]/confusion[i].sum()

# plotting
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels(['.']+all_categories,rotation=90)
ax.set_yticklabels(['.']+all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig('confusion_matrix.png')
plt.show()

# Predicting on user input
def predict(input_name,num_preds=3) : 
	print('\nGiven Name : ',input_name)
	with torch.no_grad() : 
		output=evaluate(line_to_tensor(input_name))
		# getting top N categories(languages)
		top_v,top_i=output.topk(num_preds,1,True)
		preds=[]
		for i in range(num_preds) : 
			value=top_v[0][i].item()
			category_i=top_i[0][i].item()
			print(i,'.',' Value : ',value,', Language : ',all_categories[category_i])
			preds.append([value,all_categories[category_i]])

predict('Xao')
predict('Sophia')
predict('Dre')





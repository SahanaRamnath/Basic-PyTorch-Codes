'''
--------------------------------------------------------------
SEQUENCE TO SEQUENCE MODELLING : FRENCH TO ENGLISH TRANSLATION
--------------------------------------------------------------
'''

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

##########################################################################################
# to use GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################
# Loading data and converting to embeddings(one hot vectors used)

SOS_token=0 # start of predicted sentence
EOS_token=1 # end of predicted sentence

class Lang : 
	def __init__(self,name) : 
		self.name=name
		self.word2index={}
		self.word2count={}
		self.index2word={0:"SOS",1:"EOS"}
		self.num_words=2 # currently, SOS and EOS

	def add_sentence(self,sentence) : 
		for word in sentence.split(' ') : 
			self.add_word(word)
	def add_word(self,word) : 
		if word not in self.word2index : 
			self.word2index[word]=self.num_words
			self.word2count[word]=1
			self.index2word[self.num_words]=word
			self.num_words+=1
		else : 
			self.word2count[word]+=1

def unicode_to_ascii(s) : 
	return ''.join(c for c in unicodedata.normalize('NFD',s) 
		if unicodedata.category(c)!='Mn')
def normalize_string(s) : # make i ascii,lowercase it and remove non-letter characters
	s=unicode_to_ascii(s.lower().strip())
	s=re.sub(r"([.!?])",r" \1",s)
	s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
	return s
# read from file
def read_languages(lang1,lang2,reverse=False) : 
	print('Reading from file.')

	lines=open(lang1+'-'+lang2+'.txt',encoding='utf-8').read().strip().split('\n')
	pairs=[[normalize_string(s) for s in line.split('\t')] for line in lines]

	if reverse : 
		pairs=[list(reversed(p)) for p in pairs]
		ip_lang=Lang(lang2)
		op_lang=Lang(lang1)
	else : 
		ip_lang=Lang(lang1)
		op_lang=Lang(lang2)

	return ip_lang,op_lang,pairs

# trimming dataset to sentences of max of 10 words
# so only sentences that start with below prefixes
# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re ")

max_length=10

def filter_pair(p) : 
	return (len(p[0].split(' '))<max_length and len(p[1].split(' '))<max_length)# and p[1].startswith(eng_prefixes))

def filter_pairs(pairs) : 
	return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1,lang2,reverse=False) : 
	ip_lang,op_lang,pairs=read_languages(lang1,lang2,reverse)
	print('Read ',len(pairs),' pairs.')
	pairs=filter_pairs(pairs)
	print('Filtered to ',len(pairs),' pairs.')
	for pair in pairs : 
		ip_lang.add_sentence(pair[0])
		op_lang.add_sentence(pair[1])
	print('Number of words in ',ip_lang.name,' : ',ip_lang.num_words)
	print('Number of words in ',op_lang.name,' : ',op_lang.num_words)
	return ip_lang,op_lang,pairs

ip_lang,op_lang,pairs=prepare_data('eng','fra',True)
print(random.choice(pairs))

##########################################################################################
# The Encoder-Decoder model

# encoder
class EncoderRNN(nn.Module) : 
	def __init__(self,input_size,hidden_size) : 
		super(EncoderRNN,self).__init__()

		self.hidden_size=hidden_size
		self.embedding=nn.Embedding(input_size,hidden_size)
		# number of unique words and embedding size

		self.gru=nn.GRU(hidden_size,hidden_size)
		# projecting to the same size

	def forward(self,input,hidden) : 
		embed=self.embedding(input).view(1,1,-1)
		output=embed
		output,hidden=self.gru(output,hidden)
		return output,hidden

	def init_hidden(self) : 
		return torch.zeros(1,1,self.hidden_size,device=device)

# decoder
class DecoderRNN(nn.Module) : 

	def __init__(self,hidden_size,output_size) : 
		super(DecoderRNN,self).__init__()
		self.hidden_size=hidden_size

		self.embedding=nn.Embedding(output_size,hidden_size)
		self.gru=nn.GRU(hidden_size,hidden_size)
		self.out=nn.Linear(hidden_size,output_size)
		self.softmax=nn.LogSoftmax(dim=1)

	def forward(self,input,hidden) : 
		output=self.embedding(input).view(1,1,-1) # doubt
		output=F.relu(output)
		output,hidden=self.gru(output,hidden)
		output=self.softmax(self.out(output[0]))
		return output,hidden

	def init_hidden(self) : 
		return torch.zeros(1,1,self.hidden_size,device=device)

##########################################################################################
# Attention

class Attention(nn.Module) : 
	def __init__(self,hidden_size,output_size,
		dropout_prob=0.1,max_length=max_length) : 
		super(Attention,self).__init__()

		self.hidden_size=hidden_size
		self.output_size=output_size
		self.dropout_prob=dropout_prob
		self.max_length=max_length

		self.embedding=nn.Embedding(self.output_size,self.hidden_size)
		self.attn=nn.Linear(self.hidden_size*2,self.max_length)
		self.attn_combine=nn.Linear(self.hidden_size*2,self.hidden_size)

		self.dropout=nn.Dropout(self.dropout_prob)
		self.gru=nn.GRU(self.hidden_size,self.hidden_size)
		self.out=nn.Linear(self.hidden_size,self.output_size)

	def forward(self,input,hidden,encoder_outputs) : 

		embed=self.embedding(input).view(1,1,-1)
		embed=self.dropout(embed)

		attn_weights=F.softmax(self.attn(torch.cat((embed[0],hidden[0]),1)),dim=1)
		attn_applied=torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

		output=torch.cat((embed[0],attn_applied[0]),1)
		output=self.attn_combine(output).unsqueeze(0)

		output=F.relu(output)
		output,hidden=self.gru(output,hidden)

		output=F.log_softmax(self.out(output[0]),dim=1)

		return output,hidden,attn_weights

	def init_hidden(self) : 
		return torch.zeros(1,1,self.hidden_size,device=device)


##########################################################################################
# Prepare training data

def indices_from_sentence(lang,sentence) : 
	return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang,sentence) : 
	indices=indices_from_sentence(lang,sentence)
	indices.append(EOS_token)
	return torch.tensor(indices,dtype=torch.long,device=device).view(-1,1)

def tensors_from_pair(pair) : 
	input_tensor=tensor_from_sentence(ip_lang,pair[0])
	target_tensor=tensor_from_sentence(op_lang,pair[1])
	return (input_tensor,target_tensor)

##########################################################################################
# Training
# using teacher forcing randomly based on its ratio

teacher_forcing_ratio=0.5

def train(input_tensor,target_tensor,encoder,decoder,
	encoder_optimizer,decoder_optimizer,criterion,max_length=max_length) : 

	encoder_hidden=encoder.init_hidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length=input_tensor.size(0)
	target_length=target_tensor.size(0)

	encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)

	loss=0

	for ei in range(input_length) : 
		encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)
		encoder_outputs[ei]=encoder_output[0,0]

	decoder_input=torch.tensor([[SOS_token]],device=device)

	decoder_hidden=encoder_hidden # last hidden state of encoder is first hidden state of decoder

	use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False

	if use_teacher_forcing : # target as next input to decoder
		for di in range(target_length) : 
			decoder_output,decoder_hidden,decoder_attn=decoder(decoder_input,
				decoder_hidden,encoder_outputs)
			loss+=criterion(decoder_output,target_tensor[di])
			decoder_input=target_tensor[di] # teacher forcing

	else : # it's own output as the next input to the decoder
		for di in range(target_length) : 
			decoder_output,decoder_hidden,decoder_attn=decoder(decoder_input,
				decoder_hidden,encoder_outputs)
			top_v,top_i=decoder_output.topk(1)
			decoder_input=top_i.squeeze().detach() # detach from history as input
			loss+=criterion(decoder_output,target_tensor[di])
			if decoder_input.item()==EOS_token : 
				break

	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item()/target_length

def show_loss_plot(points) : 
	plt.figure()
	fig,ax=plt.subplots()
	loc=ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
	plt.savefig('loss.png')
	plt.show()

def train_iters(encoder,decoder,num_train_steps,
	steps_per_stats=1000,steps_per_plot=100,learning_rate=0.01) : 
	
	plot_losses=[]
	print_loss_total=0
	plot_loss_total=0

	encoder_optimizer=optim.SGD(encoder.parameters(),lr=learning_rate)
	decoder_optimizer=optim.SGD(decoder.parameters(),lr=learning_rate)

	training_pairs=[tensors_from_pair(random.choice(pairs)) for i in range(num_train_steps)]
	criterion=nn.NLLLoss()

	for i in range(num_train_steps) : 

		training_pair=training_pairs[i]

		input_tensor=training_pair[0]
		target_tensor=training_pair[1]

		loss=train(input_tensor,target_tensor,encoder,decoder,
			encoder_optimizer,decoder_optimizer,criterion)

		print_loss_total+=loss
		plot_loss_total+=loss

		if i%steps_per_stats==0 : 
			print_loss_avg=print_loss_total/steps_per_stats
			print_loss_total=0
			print('Global Step : ',i,', Loss : ',print_loss_avg)

		if i>0 and i%steps_per_plot==0 :
			plot_loss_avg=plot_loss_total/steps_per_plot
			plot_loss_total=0
			plot_losses.append(plot_loss_avg)

	show_loss_plot(plot_losses)

##########################################################################################
# Evaluation
def evaluate(encoder,decoder,sentence,max_length=max_length) : 

	with torch.no_grad() : 
		input_tensor=tensor_from_sentence(ip_lang,sentence)
		input_length=input_tensor.size()[0]
		encoder_hidden=encoder.init_hidden()

		encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)

		for ei in range(input_length) : 
			encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)
			encoder_outputs[ei]=encoder_output[0,0]

	decoder_input=torch.tensor([[SOS_token]],device=device)

	decoder_hidden=encoder_hidden # last hidden state of encoder is first hidden state of decoder

	decoded_words=[]
	decoder_attns=torch.zeros(max_length,max_length) # to visualize later

	for di in range(max_length) : 
		decoder_output,decoder_hidden,decoder_attn=decoder(decoder_input,
			decoder_hidden,encoder_outputs)
		decoder_attns[di]=decoder_attn.data
		top_v,top_i=decoder_output.data.topk(1)
		if top_i.item()==EOS_token : 
			decoded_words.append('<EOS>')
			break
		else : 
			decoded_words.append(op_lang.index2word[top_i.item()])

		decoder_input=top_i.squeeze().detach()

	return decoded_words,decoder_attns[:di+1]

def evaluate_random(encoder,decoder,n=10) : 
	for i in range(n) : 
		pair=random.choice(pairs)
		print('Input : ',pair[0])
		print('Target : ',pair[1])
		output_words,attns=evaluate(encoder,decoder,pair[0])
		output_sentence=' '.join(output_words)
		print('Predicted : ',output_sentence)

hidden_size=128
encoder=EncoderRNN(ip_lang.num_words,hidden_size).to(device)
attn_decoder=Attention(hidden_size,op_lang.num_words,dropout_prob=0.1).to(device)
train_iters(encoder,attn_decoder,100000,steps_per_stats=5000)

evaluate_random(encoder,attn_decoder)

##########################################################################################
# Visualize Attention
def show_attention(input_sentence,output_words,attns,ind) : 
	fig=plt.figure()
	ax=fig.add_subplot(111)
	cax=ax.matshow(attns.numpy(),cmap='bone')
	fig.colorbar(cax)

	ax.set_xticklabels(['']+input_sentence.split(' ')+['<EOS>'],rotation=90)
	ax.set_yticklabels(['']+output_words)

	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.savefig('attention_plot_'+str(ind)+'.png')
	plt.show()

def evaluate_and_show_attention(input_sentence,ind) : 
	output_words,attns=evaluate(encoder,attn_decoder,input_sentence)
	print('Input : ',input_sentence)
	print('Predicted output : ',' '.join(output_words))
	show_attention(input_sentence,output_words,attns,ind)

evaluate_and_show_attention("je ne crains pas de mourir .",1)
evaluate_and_show_attention("elle a cinq ans de moins que moi .",2)





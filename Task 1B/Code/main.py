
# Homecoming (eYRC-2018): Task 1B
# Fruit Classification with a CNN
import torch 
import torch.nn as nn
from utils.dataset import ImageDataset,create_and_load_meta_csv_df
from model import FNet
import torchvision
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import FNet
# import required modules

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(dataset_path, debug=False, destination_path='', save=False,epochs=1):
	"""Trains model with set hyper-parameters and provide an option to save the model.

	This function should contain necessary logic to load fruits dataset and train a CNN model on it. It should accept dataset_path which will be path to the dataset directory. You should also specify an option to save the trained model with all parameters. If debug option is specified, it'll print loss and accuracy for all iterations. Returns loss and accuracy for both train and validation sets.

	Args:
		dataset_path (str): Path to the dataset folder. For example, '../Data/fruits/'.
		debug (bool, optional): Prints train, validation loss and accuracy for every iteration. Defaults to False.
		destination_path (str, optional): Destination to save the model file. Defaults to ''.
		save (bool, optional): Saves model if True. Defaults to False.

	Returns:
		loss (torch.tensor): Train loss and validation loss.
		accuracy (torch.tensor): Train accuracy and validation accuracy.
	"""
	dframe,train_data,test_data=create_and_load_meta_csv_df('./Data/fruits','./Data/dataset',randomize=None,split=.8)
	train_data=DataLoader(dataset=ImageDataset(data=train_data,transform = transforms.ToTensor()),batch_size=32)
	test_data=DataLoader(dataset=ImageDataset(test_data,transforms.ToTensor()),batch_size=32)
	costFunction=nn.CrossEntropyLoss()
	optimizer=optim.SGD(net.parameters(),lr=.001,momentum=0.9)
	#print(train_data.dataset.data)
	acc,loss=train(epochs,train_data=train_data,costFunction=costFunction,optimizer=optimizer,test_data=test_data,debug=debug,destination_path=destination_path,save=save)

	return acc,loss



	# Write your code here
	# The code must follow a similar structure
	# NOTE: Make sure you use torch.device() to use GPU if available
	
def train(epochs,train_data,costFunction,optimizer,test_data,debug,destination_path,save):
	
	net.train().to(device)
	for epoch in range(epochs):
		losses=[]
		closs=0
		for i,(image,_,idx) in enumerate(train_data,0):
			pred=net(image)
			loss=costFunction(pred,idx)
			closs+=loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if debug == True :
				losses.append(loss.item())
				print('[%d %d] loss: %.4f'%(epoch+1,i+1,closs/100))
				closs=0
		acc=accuracy(test_data)
		plt.plot(losses,label='epoch'+str(epoch))
		plt.legend(loc=1,mode='expanded')
	plt.show()
	return acc,loss.item()/100	
	
	if save == True:
		torch.save(net.state_dict(),'mode.pth')
def accuracy(test_data):
    net.eval().to(device)
    correcthit=0
    total=0
    accuracy=0
    for batches in test_data:
        data,label,idx=batches
        prediction=net(data)
        _,prediction=torch.max(prediction.data,1)
        total +=idx.size(0)
        correcthit +=(prediction==idx).sum().item()
        accuracy=(correcthit/total)*100
    print("accuracy = "+str(accuracy))
    return accuracy
	 

				
                
                
                
if __name__ == "__main__":
	net=FNet().to(device)	
	print(net)
	for parameter in net.parameters():
		print(str(parameter.data.numpy().shape)+'\n')

	acc,loss=train_model(dataset_path='./Data/fruits/',destination_path='./Data/',epochs=1,debug=True,save=True)
	print("accuracy :",acc)
	print('loss : %.4f'%loss)
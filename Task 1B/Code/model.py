import torch.nn as nn
import torch.nn.functional as F

class FNet(nn.Module):
    # Extra TODO: Comment the code with docstrings
    """Fruit Net
    """

    def __init__(self):
        # make your convolutional neural network here
        # use regularization
        # batch normalization
        super(FNet,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(3,20,5),
                                  nn.BatchNorm2d(20),
                                  nn.ReLU(),nn.Dropout2d(),
                                nn.MaxPool2d(2,2))
        self.layer2=nn.Sequential(nn.Conv2d(20,30,5), nn.BatchNorm2d(30),nn.ReLU(),nn.Dropout2d(),
                                  nn.MaxPool2d(2,2))
                                  
                                  
        self.fc1=nn.Linear(30*22*22,300)
        self.fc2=nn.Linear(300,175)
        self.fc3=nn.Linear(175,5)
        
        #pass

    def forward(self, x):
        # forward propagation
         x=self.layer1(x)
         x=self.layer2(x)
         x=x.view(-1,30*22*22)
         x=self.fc1(x)
         x=self.fc2(x)
         x=self.fc3(x)
         return x

if __name__ == "__main__":
    net = FNet()
    
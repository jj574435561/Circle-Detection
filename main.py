import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from google.colab import drive
drive.mount('/content/drive')

# Model initialization
def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)

# Data Preparation
def generate_data(num_images, size, radius, noise, b_size):
    imgs = []
    labels = []
    for i in range(num_images):
        params, img = noisy_circle(size, radius, noise)
        imgs.append(img)
        labels.append(params)
        
    imgs = np.array(imgs)
    labels = np.array(labels)
    imgs = torch.from_numpy(imgs).double()
    labels = torch.from_numpy(labels).double()
    dataset = TensorDataset(imgs, labels)
    loader = DataLoader(dataset, batch_size=b_size, 
                    num_workers=0, shuffle=False)
    return loader

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out1 = nn.Dropout(0.5)
        self.drop_out2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(256*6*6, 3)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.drop_out2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256*6*6)
        x = self.drop_out1(x)
        x = self.fc1(x)
        return x
              
def ModelTraining(model, num_train, num_val, lrate, epoch, use_cuda):
    # Using GPU
    if use_cuda and torch.cuda.is_available():
        model.cuda()
    model.apply(init_weight)

    # Preparing training and testing datasets
    train_loader = generate_data(num_train, 200, 50, 2, 10)
    test_loader = generate_data(num_val, 200, 50, 2, 10)
    
    # Loss function and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    print('Training begin')
    
    for epoch in range(epoch):
        # Training mode
        training_loss = []
        for batch_id, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 1, 200, 200))
            labels = Variable(labels)

            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            training_loss.append(loss.data)
            loss.backward()
            optimizer.step()

        # Logging training   
        print('Epoch:', epoch+1)
        print('trainerror:', sum(training_loss)/len(training_loss))
        
        # Testing mode
        model.eval()
        loss_test = []
        for batch_id, (images, labels) in enumerate(test_loader):
            images = Variable(images.view(-1, 1, 200, 200))
            labels = Variable(labels)

            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            detected = model(images)
            loss = criterion(detected, labels)
            loss_test.append(loss.data)

        # Logging testing     
        print('testerror:', sum(loss_test)/len(loss_test))
   
    # save model to google drive    
    torch.save(model.state_dict(), '/content/drive/My Drive/scaleAI/cir_dect.pt')
    return model
        
def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img
  

def find_circle(model, image):
    # Set model to inference mode
    model.eval()

    # Prepare data
    image = torch.from_numpy(np.array(image)).double()
    image = Variable(image.view(-1, 1, 200, 200)).cuda()

    # Prepare result for calculating IOU
    detected = model(image).tolist()
    detected = tuple(detected[0])

    return detected


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    ) 


def main():
    # Model Training
	# Number of Training data: 10000
	# Number of Validation data: 1000
	# learning rate: 0.001
	# Epoch: 50
	# Use GPU: True
    '''
    CNN = ConvNet().double()
    CNN = ModelTraining(CNN, 10000, 1000, 0.001, 50, True)
    '''
    
    # Testing trained model
    # Load model from google drive
    CNN = ConvNet().double().cuda()
    state_dict = torch.load('/content/drive/My Drive/scaleAI/cir_dect.pt')
    CNN.load_state_dict(state_dict)
    
    result = []
    for i in range(1000):
        label, image = noisy_circle(200, 50, 2)
        detected = find_circle(CNN, image)
        res = iou(detected, label)
        result.append(res)
    result = np.array(result)
    print('metric AP@0.7:', (result > 0.7).mean())      
    

main()

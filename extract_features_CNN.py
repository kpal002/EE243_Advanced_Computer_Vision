import os
import numpy as np
import scipy.io as sio
import glob
import torch
import numpy as np
import torchvision
from models.alexnet import alexnet
from models.vgg import vgg16
import torchvision.transforms as transforms
from PIL import Image

def main():

  alex_feature = []
  alex_label = []
  
  vgg16_feature = []
  vgg16_label = []

  transform  = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

  train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

  test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

  
  # [Problem 4 a.] IMPORT VGG16 AND ALEXNET FROM THE MODELS FOLDER WITH 
  # PRETRAINED = TRUE

  vgg16_extractor = vgg16(pretrained=True)
  vgg16_extractor.eval()
  
  alex_extractor = alexnet(pretrained=True)
  alex_extractor.eval()
  
  for idx, data in enumerate(train_data):

      image, label = data
      
      # [Problem 4 a.] OUTPUT VARIABLE F_vgg and F_alex EXPECTED TO BE THE 
      # FEATURE OF THE IMAGE OF DIMENSION (4096,) AND (256,), RESPECTIVELY.

      #feature_extractor_vgg16 = torch.nn.Sequential(*list(vgg16_extractor.children())[:-1])
      F_vgg = vgg16_extractor(image.unsqueeze(0))
      
      vgg16_feature.append(F_vgg.detach().numpy()[0])
      vgg16_label.append(label)


      #feature_extractor_alex = torch.nn.Sequential(*list(alex_extractor.children())[:-1])
      F_alex = alex_extractor(image.unsqueeze(0))
      alex_feature.append(F_alex.detach().numpy()[0])
      alex_label.append(label)

  sio.savemat('vgg16_train.mat', mdict={'feature': vgg16_feature, 'label': vgg16_label})
  sio.savemat('alexnet_train.mat', mdict={'feature': alex_feature, 'label': alex_label})
	
if __name__ == "__main__":
   main()

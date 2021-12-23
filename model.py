import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1)) 
    self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
    self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))


    self.fc1 = nn.Linear(128*8*8 ,128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, kernel_size=(2, 2))

    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, kernel_size=(2, 2))

    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, kernel_size=(2, 2))
    x = torch.flatten(x, start_dim=1)

    x = self.fc1(x)
    x = self.fc2(x)
    output = torch.softmax(x, dim=1)

    return output
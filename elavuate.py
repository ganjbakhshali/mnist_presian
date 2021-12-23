import os
import numpy as np
import cv2
import torch
import torchvision
from model import Model
from argparse import *



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--input_img",default="5.jpg", type=str)
    args = parser.parse_args()

    device=torch.device(args.device)

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0), (1)),
    ])

    img = cv2.imread(args.input_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (70, 70))

    tensor = transform(img).unsqueeze(0).to(device)

    model = Model()
    model = model.to(device)
    model.load_state_dict(torch.load("mnist_p.pth"))
    model.eval()

    pred = model(tensor)
    pred = pred.cpu().detach().numpy()
    output = np.argmax(pred)

    print(f"model prediction: {output}")
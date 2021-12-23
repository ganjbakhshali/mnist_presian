import os
import torch
import torchvision
from model import *
from torchvision import transforms
from argparse import ArgumentParser



def cal_acc(y_pred, labels):
    _,prediction = torch.max(y_pred,1)
    acc=torch.sum(prediction == labels.data, dtype = torch.float64) / len(y_pred)
    return acc

batch_size = 64
epoch = 20
lr = 0.001

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Resize((70, 70)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = torchvision.datasets.ImageFolder(root=args.data_path, transform = transform)
    train_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(0)
    train_dataset_size = int(0.8 * len(dataset))
    test_dataset_size = len(dataset) - train_dataset_size
    _, test_data = torch.utils.data.random_split(dataset, [train_dataset_size, test_dataset_size])
    test_data = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    device=torch.device(args.device)
    model = Model()
    model=model.to(device)
    model.load_state_dict(torch.load("mnist_p.pth"))
    model.eval()

    test_acc=0.0
    for img, label in test_data:

        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        test_acc += cal_acc(pred, label)

    total_acc = test_acc / len(test_data)
    print(f"test accuracy: {total_acc}")
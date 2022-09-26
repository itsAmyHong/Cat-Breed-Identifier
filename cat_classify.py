import json
from pathlib import Path

import numpy as np 
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F_t

from sklearn.preprocessing import LabelEncoder

import timm

torch.manual_seed(100)
has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

device

"""## Study image data"""

trainset_path = Path('/data/data-sets/CatBreeds/Oxford-IIIT/catonly_train/')
testset_path = Path('/data/data-sets/CatBreeds/Oxford-IIIT/catonly_test/')

# list contents in images/
cat_breeds = []
for i in trainset_path.iterdir():
    breed_tag = i.name
    if breed_tag not in cat_breeds:
        cat_breeds.append(breed_tag)

N = len(cat_breeds)
print(N)

lab_enc = LabelEncoder()
lab_enc.fit(cat_breeds)

ordered_labels = lab_enc.inverse_transform(np.arange(N)).tolist()
print(ordered_labels)

with open('ordered_labels.json', 'w') as f:
    json.dump(ordered_labels, f)

"""## Create dataset instance"""

tfcats = transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=F_t.InterpolationMode.BICUBIC),
    transforms.Resize(256, interpolation=F_t.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

tfcats_eval = transforms.Compose([
    transforms.Resize(256, interpolation=F_t.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class CustomDataset(Dataset):
    
    def __init__(self, image_folder, tf):
        self.image_folder = image_folder
        self.tf = tf

        self.samples = []
        for p in image_folder.glob('*/*.png'):
            cls_name = p.parent.name
            cls_idx = lab_enc.transform([cls_name])[0]
            self.samples.append([p, cls_idx])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_idx = self.samples[idx]
        img = Image.open(img_path)
        # apply image transform
        tsr = self.tf(img)
        return tsr, cls_idx;

trainset = CustomDataset(image_folder=trainset_path, tf=tfcats)
testset = CustomDataset(image_folder=testset_path, tf=tfcats_eval)
print(len(trainset), len(testset))

# TEST by retrieving one sample from Dataset
a, b = trainset[99]
print(a.shape, a.min(), a.max(), b)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=128,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, drop_last=False)

# TEST by retrieving one batch from DataLoader
for a, b in train_loader:
    print(a.shape, b)
    break

"""## Convolutional Neural Network """

print(timm.list_models('*mobilenet*'))

model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=N)

sum(p.numel() for p in model.parameters() if p.requires_grad)

from timm.data import resolve_data_config

config = resolve_data_config({}, model=model)

config

model = model.to(device)
# print(model)

# optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=1e-4)

# TEST by feeding a dummy input
x = torch.empty((1,3,224,224), device=device)
y = model(x)
print(x.shape, '->', y.shape)

"""## Train/Test Functions"""

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y = model(data)
        output = F.log_softmax(y, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y = model(data)
            output = F.log_softmax(y, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

"""### Start Train/Test, save model(s)"""

n_epochs = 50

acc_history = []
for i in range(1, n_epochs + 1):
    train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=i)
    acc = test(model=model, device=device, test_loader=test_loader)
    acc_history.append(acc)

model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted.pt')

"""### Visualize loss and accuray"""

import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix

def viewAccuracyGraph(accuracy_history):
    plt.plot(accuracy_history, color='orange')
    plt.title('Test Accuracy', fontsize=25)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)

viewAccuracyGraph(acc_history)


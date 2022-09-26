from pathlib import Path
import json

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_t

import random

import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

has_cuda = torch.cuda.is_available()
device = torch.device("cpu")

with open('ordered_labels.json', 'r') as f:
    ordered_labels = json.load(f)

ordered_labels

model = torch.jit.load('model_scripted.pt').to(device)
_ = model.eval()

# a folder only contains cat images
cat_images_folder = Path('./static/images')

tfcats = transforms.Compose([
    transforms.Resize(256, interpolation=F_t.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def predict_on_image(p):
    img = Image.open(p).convert('RGB')
    # img = F_t.resize(img, size=(224, 224))
    # tsr = F_t.to_tensor(img)
    tsr = tfcats(img)
    x = tsr.unsqueeze(0)
    y = model(x.to(device)).detach().cpu()
    score = F.softmax(y, dim=1)[0].numpy()
    top1_idx = score.argmax()
    breed_tag = ordered_labels[top1_idx]
    return breed_tag, score[top1_idx]

cat_image_paths = [p for p in cat_images_folder.iterdir() if p.is_file()]

p = random.choice(cat_image_paths)
img = mpimg.imread(p)
#imshow(img)
tag, prob = predict_on_image(p)


f = open('output.txt', 'w+')
f.write('It looks like the {0} cat ({1}%).'.format(tag, round(prob*100))) 
os.remove('./static/images', cat_image_paths[0])
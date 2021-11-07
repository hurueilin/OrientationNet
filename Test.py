import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models
from torchsummary import summary
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 8


class MyDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        data = []
        with open(txt_file, 'r') as file_handler:
            for line in file_handler:
                imgName = line.strip('\n')
                data.append(imgName)

        self.data = data
        self.transform = transform  
        
    def __getitem__(self, index):
        imgName = self.data[index]
        img = Image.open('data/image_test_new/'+imgName).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.data)

data_transform = transforms.Compose([
        transforms.Resize((100, 100)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
test_dataset = MyDataset(txt_file='data/name_test_new.txt', transform=data_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

# Parser for loading model
parser = ArgumentParser()
parser.add_argument("-m", "--model", help="the model(.pth) you want to load in", dest="model", default="last_model.pth")
args = parser.parse_args()

# Load model
model = torch.load(f'output/{args.model}')
summary(model, (3, 32, 32))


# ---------- Testing: save Top1 predicted label to predictions list ----------
print('======= Predictions Starts =======')
predictions = []
model.eval()
with torch.no_grad():
    total = 0
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for item in preds:
            predictions.append(item.tolist())

        total += images.size(0)
        


# Create output file
submission = []
with open('data/name_test_new.txt') as f:
     test_images = [x.strip() for x in f.readlines()]  # all the testing images

for imgName, predicted_class in zip(test_images, predictions):
    submission.append([imgName, predicted_class])

np.savetxt('output/result.txt', submission, fmt='%s')
print(f'Finish saving final predictions of {total} testing data to output/result.txt !')
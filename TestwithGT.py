import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models
from torchsummary import summary
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser


# Parser for loading model
parser = ArgumentParser()
parser.add_argument("-m", "--model", help="the model(.pth) you want to load (put in output folder)", dest="model", default="best_model.pth")
args = parser.parse_args()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32


class MyDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        data = []
        with open(txt_file, 'r') as file_handler:
            for line in file_handler:
                line = line.strip('\n')
                imgName, label = line.split()
                label = label.split('.')[0]
                data.append((imgName, label))

        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        imgName, label = self.data[index]
        # img = Image.open('data/image_train_augmented/'+imgName).convert('RGB')
        img = Image.open('data/real/image_test_new/'+imgName).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        # Convert str label to tensor
        label = torch.tensor(int(label))

        return img, label
    
    def __len__(self):
        return len(self.data)

test_transform = transforms.Compose([
        transforms.Resize((100, 100)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
# test_dataset = MyDataset(txt_file='data/syn_test_label.txt', transform=test_transform)
test_dataset = MyDataset(txt_file='data/real/image_test_new_label.txt', transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


# Load model
model = torch.load(f'output/{args.model}')
summary(model, (3, 32, 32))


# ---------- Testing: save Top1 predicted label to predictions list ----------
print('======= Predictions Starts =======')
predictions = []
running_corrects = 0
count = 0

model.eval()
for i, (images, labels) in enumerate(tqdm(test_loader, ascii=True)):
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for item in preds:
            predictions.append(item.tolist())
            count += 1

        running_corrects += torch.sum(preds == labels)

test_acc = running_corrects.double() / len(test_dataset)    
print('Test Accuracy: {:.4f}'.format(test_acc))


# Create output file
# submission = []
# # with open('data/syn_test_label.txt') as f:
# with open('data/old/image_test_new_label.txt') as f:
#     test_images = [x.strip().split()[0] for x in f.readlines()]  # all the testing images

# for imgName, predicted_class in zip(test_images, predictions):
#     submission.append([imgName, predicted_class])

# np.savetxt('output/result.txt', submission, fmt='%s')
# print(f'Finish saving final predictions of {count} testing data to output/result.txt !')
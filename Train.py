import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models
from torchsummary import summary
import numpy as np
from PIL import Image
from tqdm import tqdm
import util


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_CLASSES = 10
EPOCH = 10
BATCH_SIZE = 32
LR = 0.001


class MyTrainDataset(Dataset):
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
        img = Image.open('data/image_train_augmented/'+imgName).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        # Convert str label to tensor
        label = torch.tensor(int(label))

        return img, label
    
    def __len__(self):
        return len(self.data)


class MyValDataset(Dataset):
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
        img = Image.open('data/real/image_test_new/'+imgName).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        # Convert str label to tensor
        label = torch.tensor(int(label))

        return img, label
    
    def __len__(self):
        return len(self.data)


train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # computed from ImageNet images
    ])
val_transform = transforms.Compose([
        transforms.Resize((100, 100)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = MyTrainDataset(txt_file='data/syn_train_label.txt', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

# val_dataset = MyDataset(txt_file='data/syn_val_label.txt', transform=val_transform)
val_dataset = MyValDataset(txt_file='data/real/image_test_new_label.txt', transform=val_transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False)



# Initialize a model, and put it on the device specified.
# model = MyClassifier().to(device)
# model.device = device

# Init well-known model and modify the last FC layer
model = torchvision.models.resnet101(pretrained=True)
# print(model)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model.classifier = nn.Linear(1024, NUM_CLASSES)  # densenet
model = model.to(device)  # Send the model to GPU

summary(model, (3, 32, 32))
num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('num_of_params:', num_of_params)


# Loss and optimizer
criterion_CE = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# Train the model
curr_lr = LR
best_acc = 0
best_epoch = 0
training_loss_history, training_accuracy_history = [], []
val_loss_history, val_accuracy_history = [], []


for epoch in tqdm(range(EPOCH), ascii=True):
    # ---------- Training ---------
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    for i, (images, labels) in enumerate(tqdm(train_loader, ascii=True)):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(True):
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            cross_entropy_loss = criterion_CE(outputs, labels)
            loss = cross_entropy_loss
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Backward and optimize (only in train phase)
            loss.backward()
            optimizer.step()
           
    
        # statistics
        running_loss += loss.item() * images.size(0)  # images.size(0) means BATCH_SIZE
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    training_loss_history.append(epoch_loss)
    training_accuracy_history.append(epoch_acc.cpu())  # the element is still GPU tensor, need to convert to CPU tensor

    # print training info
    print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # Decay learning rate
    # if (epoch+1) > 5:
    #     if (epoch+1) % 3 == 0:
    #         curr_lr /= 3
    #         update_lr(optimizer, curr_lr)


    # # ---------- Validation ----------
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    softmax = nn.Softmax(dim=-1)  # Define Softmax function

    for i, (images, labels) in enumerate(tqdm(val_loader, ascii=True)):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(False):         
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            loss = criterion_CE(outputs, labels)

        # statistics
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels)


    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = running_corrects.double() / len(val_dataset)
    val_loss_history.append(epoch_loss)
    val_accuracy_history.append(epoch_acc.cpu())  # the element is still GPU tensor, need to convert to CPU tensor

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch
        torch.save(model, 'output/best_model.pth')
    
    # print validation info
    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


# Output the training info cruves
util.save_loss_history(training_loss_history, val_loss_history, EPOCH)
util.save_accuracy_history(training_accuracy_history, val_accuracy_history, EPOCH)
print('Best epoch:', best_epoch)
print('Best accuracy in validation:', best_acc)

print('Finish training. The model is saved in output folder.')

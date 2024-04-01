get_ipython().system('pip install datasets')
from datasets import load_dataset
dataset = load_dataset("valhalla/emoji-dataset")

# import statements
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
import random
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
from torchvision.transforms import functional as TF
from datasets import DatasetDict
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


dataset

##Dataset
# 
# * The dataset is a subset of the emoji dataset on Hugging face. I have used the 'text' to make this subset.
# * I searched for keyword 'face' in the text to get emoji faces. This gave me 204 datapoints.
# * To increase the size of the dataset, I used data augmentation. A random set of transformations was added to every image in the dataset, thus increasing the size of dataset to 408 points.
# * The dataset was then broken into train, validation and test sets in a ratio of 60/20/20.

# creating. a subset that will only have face related emojis
subset = dataset.filter(lambda example: 'face' in example['text'])

subset
subset['train'][0]['image']


train = subset['train']

transformed_subset = []
def transform_example(example):
    image = example['image']
    image = image.convert('RGB').resize((64, 64))
    image_tensor = ToTensor()(image)

    return transformed_subset.append(image_tensor)

subset.map(transform_example, batched=False, remove_columns=['image', 'text'])

# data augmentation

augmented = []

def augment (tensor):
  random_number = random.randint(1, 3)

  if random_number == 1:
    data_transforms = Compose([
        Resize((64, 64)),  # Resize all images to 64x64
        RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # Randomly crop and resize
        ToTensor(),  # Convert images to PyTorch tensors
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # Normalize tensors
    ])

  elif random_number ==2:
    data_transforms = Compose([
        Resize((64, 64)),  # Resize all images to 64x64
        RandomHorizontalFlip(),  # Randomly flip images horizontally
        RandomRotation(degrees=15),  # Randomly rotate images by +/- 15 degrees
        ToTensor(),  # Convert images to PyTorch tensors
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # Normalize tensors
    ])

  else:
    data_transforms = Compose([
        Resize((64, 64)),  # Resize all images to 64x64
        RandomHorizontalFlip(),  # Randomly flip images horizontally
        RandomRotation(degrees=15),  # Randomly rotate images by +/- 15 degrees
        RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # Randomly crop and resize
        ToTensor(),  # Convert images to PyTorch tensors
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # Normalize tensors
    ])


  image = TF.to_pil_image(tensor)
  aug_image = data_transforms(image)
  augmented.append(aug_image)

for image in transformed_subset:
  augment(image)


combined_dataset = transformed_subset+augmented
random.shuffle(combined_dataset)


# Splitting the dataset into train, validation, and test sets
train_size = int(0.6 * len(combined_dataset))
test_size = int(0.2 * len(combined_dataset))
val_size = len(combined_dataset) - train_size - test_size  # Ensuring the remainder is assigned to val_size

# split
train_dataset, val_dataset, test_dataset = combined_dataset[:train_size], combined_dataset[train_size:train_size+val_size], combined_dataset[train_size+val_size:]

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

train_loader


# Architecture
# 
# * The autoencoder has two components - encoder and decoder.
# * The encoder has 4 CNN layers. All layers use ReLU activation, stride = 2 and padding = 1. The input and output sizes are:
# > * 1st Convolution layer - 3 i/p and 32 o/p
# > * 2nd Convolution layer - 32 i/p and 64 o/p
# > * 3rd Convolution layer - 64 i/p and 128 o/p
# > * 4th Convolution layer - 128 i/p and 256 o/p
# 
# * The decoder has 4 Transpose CNN layers that mirror the CNN layers of the encoder model. All layers use ReLU activation, stride = 2 and padding = 1. The input and output sizes are:
# > * 1st Transpose Convolution layer - 256 i/p and 128 o/p
# > * 2nd Transpose Convolution layer - 128 i/p and 64 o/p
# > * 3rd Transpose Convolution layer - 64 i/p and 32 o/p
# > * 4th Transpose Convolution layer - 32 i/p and 3 o/p

# Design Choices
# 
# * I experimented with different number of CNN layers. While anything less than 3 layers gave a higher loss in testing sets. When the layers was more than 5, the colab file would crash because of the overload with 80 epochs, which I found to work best.
# 
# * Thus, I tried running different combinations of train, test and validation sets on the layers. 4 layers was the best in most cases when I compared the loss v/s the training time.
# 
# * I chose the CNN architecture since I was dealing with images. Using linear neural networks would require complex network and may make the computations very difficult.  


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # [batch, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [batch, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [batch, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # [batch, 256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Hyperparameters
# * Learning rate - 0.01
# * Epochs - 80
# * Criterion - Mean Squared Error
# * Optimizer - Adam
# * Batch size - 32


model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):

    train_losses = []
    val_losses = []

    # training
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch)  # Forward pass
            loss = criterion(outputs, batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                loss = criterion(outputs, batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch+1) % 10 ==0:
          print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    return train_losses, val_losses

train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=80)


# Learning Curves

def plot_curves(train_losses,val_losses,ylab):
  if ylab == 'accuracy':
    plt.title('Classification Accuracy')
    plt.plot(train_losses, label='Training Accuracy')
    plt.plot(val_losses, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel(ylab)
  else:
    plt.title('Learning Curves')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel(ylab)
  plt.legend()
  plt.show()


plot_curves(train_losses,val_losses,'loss')


# Average test loss

model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch)
        for tensor in batch:
          image = TF.to_pil_image(tensor)
          image
        #print(outputs)
        loss = criterion(outputs, batch)
        test_loss += loss.item()

average_test_loss = test_loss / len(test_loader)
print(f'Average Test Loss: {average_test_loss}')


# ## Discussion
# 
# * The change in learning rate significantly impacts the average loss.
# * The encoder - decoder model with a simpler CNN structure did not work well.
# * Data augmentation helped increase the generalizability of the model. Initially the dataset was only 204 images. I increased it by 100% to 408 images.
# * Increasing the number of epochs and decreasing the learning rate might lead to a more robust and accurate model.

# # Question 2

# ## Dataset
# 
# * I made the dataset using the 'text' description of the emoji dataset.
# * The dataset consists of 2 classes - male and female.
# * The male class collected all points that had words like father, male, boy and man in it's text.
# * The female class, on the other hand, had words like mother, female, girl and woman.
# * Overall, the dataset has 841 datapoints. 443 belong to the male class while 398 was from the female class. Thus the dataset has a good balance.
# 
# 

# In[ ]:


# checking if datapoint is male
def is_male(text):
  male = ['father', 'male', 'boy','man']
  words = text.split()
  for word in words:
    if word in male:
      return True
  return False


# In[ ]:


# checking if datapoint is female
def is_female(text):
  female = ['mother', 'female', 'girl','woman']
  words = text.split()
  for word in words:
    if word in female:
      return True
  return False


# In[ ]:


# two datasets - male and female
male_subset = []
female_subset = []

for example in dataset['train']:
    if is_male(example['text']):
        male_subset.append([example['image'], 0])  # 0 for male
    elif is_female(example['text']):
        female_subset.append([example['image'], 1])  # 1 for female

combined_dataset = male_subset + female_subset


# In[ ]:


def trans(example):
    image = example.convert('RGB').resize((64, 64))
    image_tensor = ToTensor()(image)
    return image_tensor


# In[ ]:


for tup in combined_dataset:
  img = tup[0]
  image_tensor = trans(img)
  tup[0] = image_tensor


# In[ ]:


# Set the seed for reproducibility
random_seed = 42
random.seed(random_seed)
random.shuffle(combined_dataset)


# In[ ]:


from datasets import DatasetDict

# Splitting the dataset into train, validation, and test sets
train_size = int(0.6 * len(combined_dataset))
test_size = int(0.2 * len(combined_dataset))
val_size = len(combined_dataset) - train_size - test_size

# Perform the split
train_dataset, val_dataset, test_dataset = combined_dataset[:train_size], combined_dataset[train_size:train_size+val_size], combined_dataset[train_size+val_size:]

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")


# ## Classifier
# 
# * The encoder and decoder blocks are the same architecture as part 1.
# * In addition, I also added a classifier block that classifies the emojis in their respective groups. This block consist of a simple linear neural network.
# * Architecture of classifier:
# > * The first layer will take o/p of the last decoder layer, thus it has 256 nodes. There is a 50% dropout here to reduce the complexity. It uses ReLU activation and output is 512 dimensions.
# > * The second layer also uses ReLU and 50% dropout. However, this layer has an input size of 512 and output is of size 128.
# > * The third layer is also similar. It's input is of size 28 and output size is 2. Softmax is used for the classification.
# 
# * Hyper parameters:
# >* Reconstruction criterion = MSE Loss
# >* Learning rate = 0.01
# >* Lambda = 0.4
# >* Optimizer = Adam
# >* Epochs = 10
# >* Batch Size = 32

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
class AutoencoderWithClassifier(nn.Module):
    def __init__(self):
        super(AutoencoderWithClassifier, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # [batch, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [batch, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [batch, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # [batch, 256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded_flat)
        return decoded, classification


## MSE and loss curves

from torch.utils.data import DataLoader
train_losses_new = []
val_losses_new = []


model = AutoencoderWithClassifier()
reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lambda_value = 0.4 # Hyperparameter to balance the two losses

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []

def train_and_plot(model, train_loader, val_loader, reconstruction_criterion, classification_criterion, optimizer, lambda_value, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            reconstructed, classifications = model(images)
            reconstruction_loss = reconstruction_criterion(reconstructed, images)
            classification_loss = classification_criterion(classifications, labels)
            loss = reconstruction_loss + lambda_value * classification_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(classifications.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy.append(100 * correct_train / total_train)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                reconstructed, classifications = model(images)
                reconstruction_loss = reconstruction_criterion(reconstructed, images)
                classification_loss = classification_criterion(classifications, labels)
                loss = reconstruction_loss + lambda_value * classification_loss
                total_val_loss += loss.item()
                _, predicted = torch.max(classifications.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy.append(100 * correct_val / total_val)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train Accuracy: {100 * correct_train / total_train:.2f}%, Validation Accuracy: {100 * correct_val / total_val:.2f}%')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


train_and_plot(model, train_loader, val_loader, reconstruction_criterion, classification_criterion, optimizer, lambda_value, num_epochs=10)

# plotting curves
plot_curves(train_losses, val_losses,'loss')


# Classification Accuracy and Curves for Classification Accuracy (increases as the number of epochs increase)


def test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Classification Accuracy: {accuracy}%')


test_accuracy(model, test_loader, device)
plot_curves(train_accuracy,val_accuracy,'accuracy')


# ## Impact of classifier
# 
# I changed the lamda values to see the impact of the classifier:
# 
# **Without Classifier (Lambda = 0):**
# * lambda = 0.0 :
#   * Classification Accuracy = 54.16%
#   * Train Loss: 0.0177
#   * Validation Loss: 0.0188
#   * Train Accuracy: 51.59%
#   * Validation Accuracy: 49.11%
# 
# **With Classifier (Lambda > 0)**
# * lambda = 0.1 :
#   * Classification Accuracy = 92.85%
#   * Train Loss: 0.0365
#   * Validation Loss: 0.0741
#   * Train Accuracy: 93.65%
#   * Validation Accuracy: 84.62%
# * lambda = 0.2 :
#   * Classification Accuracy = 94.64%
#   * Train Loss: 0.0652
#   * Validation Loss: 0.1162
#   * Train Accuracy: 91.47%
#   * Validation Accuracy: 78.11%
# * lambda = 0.3 :
#   * Classification Accuracy = 88.69%
#   * Train Loss: 0.0880
#   * Validation Loss: 0.1301
#   * Train Accuracy: 91.43%
#   * Validation Accuracy: 86.98%
# * lambda = 0.4 :
#   * Classification Accuracy = 93.45%
#   * Train Loss: 0.0634
#   * Validation Loss: 0.0678
#   * Train Accuracy: 96.03%
#   * Validation Accuracy: 96.45%
# * lambda = 0.5 :
#   * Train Loss: 0.1436
#   * Validation Loss: 0.1560
#   * Train Accuracy: 90.28%
#   * Validation Accuracy: 87.57%
#   * Classification Accuracy = 94.64%
# 
# **Performance Change**
# 
# * Adding Classification as an auxilliary task impacts the learning abilities for the encoder/decoder itself due to the fact that the loss being used to update weights is calculated by including weighted loss for the classification task (by controlling lambda) instead of updating weights just based on the actual loss for encoder/decoder.
# 
# * As seen in Q-2d, increasing the lambda value benefits the classification task by achieving high classification accuracies; however, the performance of the Encoder/Decoder decreases as the models learns to adapt strongly to the classification task, based on the set lambda value.
# 
# * However, with lambda = 0.4, the loss seem to have decreased for both taining and validation data, indicating that this lambda value complements the autoencoder to learn features accurately, and also indicates the balance across the reconstruction task and the classification task. Accordingly, this lambda value can be considered optimal for this architecture of Autoencoder w/ classification.
# 

# ##  Speculation and Recommendation
# As discussed earlier, adding a classification task with an optimal lambda value increases the performance of the autoencoder (reflected in reduced loss). This is due to the fact that:
# 1. the addition of a classification task can encourage the autoencoder to learn more meaningful and generalizable features in its latent space. This is because the features that are useful for classification are likely to be informative for reconstruction as well.
# 
# 2. adding a classification task can act as a form of regularization, preventing the autoencoder from overfitting to the training data and improving its ability to generalize to unseen examples.
# 
# 3. foundation of the classification task is to accurately learn the features of the image to achieve a good accuracy. Accordingly, adding the classification loss to the overall loss for weight updation (at an optimal lambda) can improve the depth of feature understanding of the autoencoder.
# 
# However, if the lambda value is not optimal, it may decrease the performance of the system as observed in 2d.
# 
# **Validation of speculation:**
# * Perform analysis of the autoencoder with lambda = 0 (w/o classification) and lambda > 0 (with classification) to validate the speculation. If the latent representation is improved as stated in point 1 above, it justifies the claim.
# 
# * An experimental analysis can be done by using:
#   1. Autoencoder w/ optimal regularization and
#   2. Autoencoder w/ classification (optimal lambda value)
# 
#   and compare the results to determine if the results are close as expected. If the results are near to each other, it can bolster the second point 2 discussed above that mentions classifications acts as a regularizer for autoencoder architecture.


# Combining Emojis using Vector Arithmatic



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import random

from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = load_dataset('valhalla/emoji-dataset', split='train')

subset = dataset.filter(lambda example: 'face' in example['text'])
c=0
transformed_subset = []

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

for i in subset:
  transformed_subset.append(transform(i['image']))

from datasets import DatasetDict

train_loader = DataLoader(transformed_subset, batch_size=1)
img = []
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




def train_and_validate(model, train_loader, criterion, optimizer, num_epochs=80):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch)  # Forward pass
            loss = criterion(outputs, batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        if (epoch +1) % 10 == 0:
          print("Epoch: " + str(epoch+1) + "  Loss: " + str(loss.item()))

train_and_validate(model, train_loader, criterion, optimizer, num_epochs=100)


# ## Feature Selection, Vector Arithmetic, Resulting Image
# **Features Selected:**
# 1.   **Sunglasses** from "similing face with sunglasses"
# 2.   **Smile** from "slightly smiling face"
# 3.   **Tongue** from "face with stuck out tongue"
# 
# **Vector Arithmetic:**
# * nf_latent = encoder : "similing face with sunglasses"
# * sf_latent = encoder : "slightly smiling face"
# * tf_latent = encoder : "face with stuck out tongue"
# 
# **combined_latent** = nf_latent - sf_latent + tf_latent
# 
# reconstructed_image = decoder : combined_latent
# 
# **Resulting Image:**
# * Displayed below the code.
# 
# 
# 
# 
# 
# 
# 

dataset = load_dataset('valhalla/emoji-dataset', split='train')

nf = 0
sf = 0
tf = 0
pl = 0
mi = 0
ar = 0

for i in range(len(dataset)):
  if dataset[i]['text'] == "smiling face with sunglasses":
    nf = i
  elif dataset[i]['text'] == "slightly smiling face":
    sf = i
  elif dataset[i]['text'] == "face with stuck out tongue":
    tf = i
  elif dataset[i]['text'] == "heavy plus sign":
    pl = i
  elif dataset[i]['text'] == "heavy minus sign":
    mi = i
  elif dataset[i]['text'] == "black rightwards arrow":
    ar = i

  if nf != 0 and sf != 0 and tf != 0 and pl != 0 and mi != 0 and ar != 0:
    break

nf = transform(dataset[nf]['image'])
sf = transform(dataset[sf]['image'])
tf = transform(dataset[tf]['image'])
pl = transform(dataset[pl]['image'])
mi = transform(dataset[mi]['image'])
ar = transform(dataset[ar]['image'])

nf_latent = model.encoder(nf)
sf_latent = model.encoder(sf)
tf_latent = model.encoder(tf)

# Perform vector arithmetic to get the combined latent vector
combined_latent = nf_latent -  sf_latent + tf_latent

reconstructed_image = model.decoder(combined_latent)
#plt.figure(figsize=(16, 4))


output_numpy = nf.squeeze().permute(1, 2, 0).detach().cpu().numpy()
output_numpy = output_numpy * 0.5 + 0.5
plt.subplot(1, 7, 1)
plt.imshow(output_numpy)
plt.axis('off')

output_numpy = mi.squeeze().permute(1, 2, 0).detach().cpu().numpy()
output_numpy = output_numpy * 0.5 + 0.5
plt.subplot(1, 7, 2)
plt.imshow(output_numpy)
plt.axis('off')

output_numpy = sf.squeeze().permute(1, 2, 0).detach().cpu().numpy()
output_numpy = output_numpy * 0.5 + 0.5
plt.subplot(1, 7, 3)
plt.imshow(output_numpy)
plt.axis('off')

output_numpy = pl.squeeze().permute(1, 2, 0).detach().cpu().numpy()
output_numpy = output_numpy * 0.5 + 0.5
plt.subplot(1, 7, 4)
plt.imshow(output_numpy)
plt.axis('off')

output_numpy = tf.squeeze().permute(1, 2, 0).detach().cpu().numpy()
output_numpy = output_numpy * 0.5 + 0.5
plt.subplot(1, 7, 5)
plt.imshow(output_numpy)
plt.axis('off')

output_numpy = ar.squeeze().permute(1, 2, 0).detach().cpu().numpy()
output_numpy = output_numpy * 0.5 + 0.5
plt.subplot(1, 7, 6)
plt.imshow(output_numpy)
plt.axis('off')

# Denormalize and display the reconstructed image
output_numpy = reconstructed_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
plt.subplot(1, 7,7)
plt.imshow(output_numpy)
plt.axis('off')
plt.show()


# ## Qualitative Evaluation
# 
# * The image appears near to what was expected, like the image has sunglasses and the tongue.
# * The positioning of the toungue and sunglasses appear near to the expectation.
# * Due to the addition of "EYES" in the "tongue emoji", the eyes are a bit visible over the glasses.
# *  The image appears a bit blurry relating to the inefficiency of the encoder. This indicates the requirement of a more complex architecture and a significant training duration requirement.
# *   The color accuracy is not exactly accurate. Also the brightness of certain sections of the image are not as expected, like the forehead appears to be more brighter than required.
# 
# 

# ## Image Quality Improvement
# 
# * **Architecture Design:** Use more convolution layers to capture more details.
# * **Training Data / Data Augmentation:** Improve the quality of Training Data or increase the Training Data via Data Augmentation for the model to capture more features.
# * **Loss Function:** Change the Loss Function to update weight more suitably for the task of image generation using latent arithmetic.  
# * **Regularization:** Use regularization to improve the quality of the trained model, making it more generalizable across the different emoji expressions in the training data.
# * **Hyperparameter Tuning:** Using lower learning rate and high number of epochs to decrease the overall training loss.
# * **Post-processing:** Add post-processing steps on the generated image by denoising and sharpening it.
# 

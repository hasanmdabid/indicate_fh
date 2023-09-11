
import numpy as np
#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code. 
np.random.seed(1000)

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from skimage.transform import resize  
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from keras.utils import normalize
from sklearn.model_selection import train_test_split, KFold
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import equalize_adapthist
import tensorflow as tf
import platform

#Iterate through all images in Parasitized folder, resize to 64 x 64
#Then save as numpy array with name 'dataset'
#Set the label to this as 0


def check_gpu():

    if 'linux' in platform.platform().lower():
        print("Check GPU...")
        if len(tf.config.list_physical_devices('GPU')) == 0:
            print("GPU is not available!")
            quit()

        print("GPU is available!")

check_gpu()

image_directory = "/home/abidhasan/Documents/Indicate_FH/data/"
SIZE = 512
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.




not_effected = os.listdir(image_directory + 'not_effected/')
for i, image_name in enumerate(not_effected):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'not_effected/' + image_name) # Reading all the images using opencv as BGR format. 
        if (image is None):  # Checking is any image is an None type object or Not. If it is, then next imge is continued.
                continue
        # converting BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = equalize_adapthist(image, kernel_size=None, clip_limit=0.1, nbins=256) # Applying the CLAHE to increase the contrast of the image
        image = img_as_ubyte(resize(image, (SIZE, SIZE), anti_aliasing=True)) 
        # Resizing the Image into 256 * 256 
        # Converting an image to unsigned byte format, with values in [0, 255].
        dataset.append(np.array(image))
        label.append(0)
dataset = np.array(dataset)
print(dataset.shape)
        
#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

effected = os.listdir(image_directory + 'effected/')
for i, image_name in enumerate(effected):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'effected/' + image_name)
        if (image is None):  # Checking is any image is an None type object or Not. If it is, then next imge is continued.
                continue
        # converting BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = equalize_adapthist(image, kernel_size=None, clip_limit=0.1, nbins=256) # Applying the CLAHE to increase the contrast of the image
        image = img_as_ubyte(resize(image, (SIZE, SIZE), anti_aliasing=True))
        # Resizing the Image into 256 * 256 
        # Converting an image to unsigned byte format, with values in [0, 255].
        dataset.append(np.array(image))
        label.append(1)


dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape)
print(label.shape)

print(np.unique(label, return_counts=True))

    
###############################################################    
    
 ### Split the dataset

# split the dataset into training and testing dataset.
# 1. Training data: 80%
# 2. Testing data: 20%

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

#Without scaling (normalize) the training may not converge. 
#Normalization is a rescaling of the data from the original range 
#so that all values are within the range of 0 and 1.

X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



#Apply CNN
# ### Build the model

#############################################################
###2 conv and pool layers. with some normalization and drops in between.

INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)

##############################################################################################################
# Calling the K fold
# Define the K-fold Cross Validator
kfold = KFold(n_splits=2, shuffle=True)

###############################################################################################################

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# K-fold Cross Validation model evaluation
 
fold_no = 1

for train, test in kfold.split(dataset, label):
    # Define the model architectur
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2)) 
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # fit network

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X_train, 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 50,      #Changed to 3 from 50 for testing purposes.
                         validation_data=(X_test, y_test),
                         validation_split = 0.1,
                         shuffle = False
                      #   callbacks=callbacks
                     )

    # Generate generalization metrics
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)
max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.savefig("/home/abidhasan/Documents/Indicate_fh/filtered_image/CNN_TRAIN_VS_VALIDATION_LOSS.png")


#Save the model
model.save('/home/abidhasan/Documents/Indicate_FH/saved_model/cnn_classification.h5')


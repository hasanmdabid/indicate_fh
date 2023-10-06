
import numpy as np
#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code. 
np.random.seed(1000)

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from skimage.transform import resize  
from keras.utils import normalize
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from skimage import img_as_ubyte
from skimage.exposure import equalize_adapthist
import tensorflow as tf
import platform
import logging
logging.getLogger('tensorflow').disabled = True
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
from models import *
from save_results import saveResultsCSV

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
modelpath = '/home/abidhasan/Documents/Indicate_FH/saved_model'
SIZE = 256
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
        #image = equalize_adapthist(image, kernel_size=None, clip_limit=0.1, nbins=256) # Applying the CLAHE to increase the contrast of the image
        image = img_as_ubyte(resize(image, (SIZE, SIZE), anti_aliasing=True)) 
        # Resizing the Image into 256 * 256 
        # Converting an image to unsigned byte format, with values in [0, 255].
        dataset.append(np.array(image))
        label.append(0)

        
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
        #image = equalize_adapthist(image, kernel_size=None, clip_limit=0.1, nbins=256) # Applying the CLAHE to increase the contrast of the image
        image = img_as_ubyte(resize(image, (SIZE, SIZE), anti_aliasing=True))
        # Resizing the Image into 256 * 256 
        # Converting an image to unsigned byte format, with values in [0, 255].
        dataset.append(np.array(image))
        label.append(1)


dataset = np.array(dataset)
print('Shape of dataset', dataset.shape)
label = np.array(label)

print('Shape of the labels', label.shape)

print('Count of the labels in dataset', np.unique(label, return_counts=True))

"""
 
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
print('Maximume and minimume value in testset =', np.max(X_test), np.min(X_test))


print(f"Shape of training set: {X_train.shape}")
print(f'Shape of testing set: {X_test.shape}')
print(f'Shape of train label:{y_train.shape}')
print('Count of the train dataset labels', np.unique(y_train, return_counts=True))
print(f'Shape of test label:{y_test.shape}')
print('Count of the test dataset labels', np.unique(y_test, return_counts=True))
""" 


##############################################################################################################
# Calling the K fold
# Define the K-fold Cross Validator
num_folds = 5
fold_method = input('Do you want to use k-fold of stratified k_fold cross validation? (kf/skf)')
if fold_method.lower() == 'kf':
    kfold = KFold(n_splits=num_folds, shuffle=True)
elif fold_method.lower() == 'skf':
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
else:
    print('You did not specify the K-fold cross validation method correctly. Please select the correct method.')
###############################################################################################################

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
f1_score_per_fold = []
best_model = None
best_macro_f1 = 0.0
average_f1_scores = []
# K-fold Cross Validation model evaluation
# Initializing the model parameters 
method = 'No_aug_no_weights'
input_shape = (SIZE, SIZE, 3)
dropout_rt = 0.4
fold_no = 1
nr_epochs = 100
batch_size = 64
batch_size = 32
# Parameters to initialize the google_vit model
num_layers = 12
embed_dim = 768
num_heads = 12
ff_dim = 3072

model_names = ['my_model', 'VGG3', 'resnet50']

for model_name in model_names:
    #Create the model
    if model_name == 'my_model':
        model = my_model(input_shape, dropout_rt)
    elif model_name == 'vit_model':
        model = create_vit_model(input_shape, num_layers, embed_dim, num_heads, ff_dim, dropout_rt)
    elif model_name == 'VGG3':
        model = VGG3(input_shape, dropout_rt)
    elif model_name == 'resnet50':
        model = resnet50(input_shape, dropout_rt)

    for train, test in kfold.split(dataset, label):
        # fit network
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        X_train = dataset[train]
        X_test = dataset[test]
        y_train = label[train]
        y_test = label[test]

        # Fit data to model
        #history = model.fit(datagen.flow(X_train, y_train, batch_size= batch_size), steps_per_epoch=len(X_train) // batch_size, verbose = 1,  epochs=100, validation_data=(X_test, y_test), shuffle = False)
        history = model.fit(X_train, y_train, batch_size = batch_size, verbose = 1, epochs = nr_epochs, validation_data=(X_test, y_test), validation_split = 0.1, shuffle = False)

        # Generate generalization metrics
        scores = model.evaluate(X_test, y_test, verbose=1)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        # Calculate F1 score on validation data for the current fold
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        macro_f1 = f1_score(y_test, y_pred_binary, average='macro')
        f1_score_per_fold.append(macro_f1 *100)
        # Check if this fold's model has the best macro F1 score
        if macro_f1 > best_macro_f1:
            best_model = model
            best_macro_f1 = macro_f1

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print(f"Model Name= {model_name}")
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(f1_score_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - F1_Score_macro: {f1_score_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Avg_Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Avg_F1_Score_Macro: {np.mean(f1_score_per_fold)}')
    print('------------------------------------------------------------------------')
    print('Maximume Macro F1 scores for all folds:')
    print(f'> Max_F1_Score_Macro: {np.max(f1_score_per_fold)}')
    print('------------------------------------------------------------------------')
    avg_accuracy = np.mean(acc_per_fold)
    std_avg_accuracy = np.std(acc_per_fold)
    max_f1_score_macro  = np.max(f1_score_per_fold)
    avg_f1_score_macro = np.mean(f1_score_per_fold)
    std_f1_score_macro = np.std(f1_score_per_fold)
    saveResultsCSV(method, model_name, nr_epochs, batch_size, avg_accuracy, std_avg_accuracy,  avg_f1_score_macro, std_f1_score_macro, max_f1_score_macro)
    # Save the best model
    best_model.save(modelpath+"/"+f"{model_name}_Kfold_best_model.h5")

"""
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
plt.savefig("/home/abidhasan/Documents/Indicate_FH/performance_figures/my_model_kfold_CNN_TRAIN_VS_VALIDATION_LOSS.png")


# Save the best model
best_model.save(modelpath+'/my_model_Kfold_best_model.h5')
"""


"""
This script will generate handcrafted features form the CLE images
Script Author: Md Abid Hasan
Project: Indicate_FH
Date: 12 July 2023

"""
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.exposure import equalize_adapthist

import numpy as np
from scipy import ndimage as nd
from skimage import filters
import pandas as pd
from skimage.transform import resize
import pandas as pd
import cv2
import os
from tqdm import tqdm
import pickle


"""
Feature based segmentation using Random Forest
Demonstration using multiple training images

STEP 1: READ all the Images (Both Effected and Not effected) and EXTRACT FEATURES 

STEP 2: Add labels at the last column of the dataframe 

STEP 3: GET DATA READY FOR RANDOM FOREST (or other classifier)

STEP 4: DEFINE THE CLASSIFIER AND FIT THE MODEL USING TRAINING DATA

STEP 5: CHECK ACCURACY OF THE MODEL

STEP 6: SAVE MODEL FOR FUTURE USE

STEP 7: MAKE PREDICTION ON NEW IMAGES

"""

############################ Reading aLL the image ###############

image_size = 256
effected_img_path = "/home/abidhasan/Documents/Indicate_fh/data/effected/"
not_effected_img_path = "/home/abidhasan/Documents/Indicate_fh/data/not_effected/"
#effected_dataset = pd.DataFrame()  # Dataframe to capture not_effected image features
#not_effected_dataset = pd.DataFrame()
# Dataframe to capture not_effected image features


def read_image(image_path, label):
    dataset = pd.DataFrame() 
    for image in tqdm(os.listdir(image_path)):  # iterate through each file
        df = (pd.DataFrame())  # Temporary data frame to capture information for each loop.
        if image.endswith(".png"):
            image = cv2.imread(image_path + image, cv2.IMREAD_GRAYSCALE)  # Read images
            # Reset dataframe to blank after each loop.
            if (image is None):  # Checking is any image is an None type object or Not. If it is, then next imge is continued.
                continue
            image = cv2.resize(image, (image_size, image_size))
            #image = resize(image, (image_size, image_size), anti_aliasing=True)

        ################################################################
        # START ADDING DATA TO THE DATAFRAME

        #### Add pixel values to the data frame
        pixel_values = image.reshape(-1)
        df["Original_Image"] = pixel_values  # Pixel value itself as a feature

        #############################################################################
        # Generate Gabor features

        #####      Generate Gabor features
        num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):  # Define number of thetas
           theta = theta / 4.0 * np.pi
           for sigma in (1, 3):  # Sigma with 1 and 3
               for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                   for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
                       gabor_label = "Gabor" + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                    # print(gabor_label)
                       ksize = 9
                       kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                       kernels.append(kernel)
                       # Now filter the image and add values to a new column
                       fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                       #fimg = fimg.astype(np.float32)   # normalized image
                       filtered_img = fimg.reshape(-1)
                       df[ gabor_label] = filtered_img  # Labels columns as Gabor1, Gabor2, etc.
                       # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                       num += 1  # Increment for gabor column label

        # Applying the Contrast limited adaptive histogram equalization to increase the contrast of the image
        CLAHE = equalize_adapthist(image, kernel_size=None, clip_limit=0.1, nbins=256)
        #CLAHE = CLAHE.astype(np.float32)   # normalized image
        CLAHE1 = CLAHE.reshape(-1)
        df["CLAHE"] = CLAHE1

        # Appling the Gaussian filter with a Gaussian filter Scipy
        gaussian = nd.gaussian_filter(image, sigma=3)
        #gaussian = gaussian.astype(np.float32) 
        gaussian1 = gaussian.reshape(-1)
        df["gaussian"] = gaussian1

        # Appling the Mean filter with Scipy-----------------------------------------------------------------------------
        median = nd.median_filter(image, size=4)
        #median = median.astype(np.float32) 
        median1 = median.reshape(-1)
        df["median"] = median1

        # Appling the PREWITT filter with

        prewitt = filters.prewitt(image)
        #prewitt = prewitt.astype(np.float32) 
        prewitt1 = prewitt.reshape(-1)
        df["prewitt"] = prewitt1

        ## Appling the Sobel filter with Scipy-----------------------------------------------------------------------------

        sobel = filters.sobel(image)
        #sobel = sobel.astype(np.float32) 
        sobel1 = sobel.reshape(-1)
        df["sobel"] = sobel1

        farid = filters.farid(image, mode="reflect")
        #farid = farid.astype(np.float32) 
        farid1 = farid.reshape(-1)
        df["farid"] = farid1

        butter = filters.butterworth(image, cutoff_frequency_ratio=0.005,high_pass=True,order=7.0,squared_butterworth=True,npad=0)
        #butter = butter.astype(np.float32) 
        butter1 = butter.reshape(-1)
        df["butter"] = butter1

        robert = filters.roberts(image, mask=None)
        #robert = robert.astype(np.float32) 
        robert1 = robert.reshape(-1)
        df["robbert"] = robert1
            #SCHARR
        edge_scharr = filters.scharr(image)
        #edge_scharr = edge_scharr.astype(np.float32) 
        edge_scharr1 = edge_scharr.reshape(-1)
        df['Scharr'] = edge_scharr1
    
        # Applying Non Local Median filter--------------------------------------------------------------------------------
        sigma_st = np.mean(estimate_sigma(image))
        nlm = denoise_nl_means(image, h=1.7 * sigma_st, fast_mode=True, patch_size=5, patch_distance=3)
        #nlm = nlm.astype(np.float32) 
        nlm1 = nlm.reshape(-1)
        df["NLM"] = nlm1
        
        #Adding the labels at the last o the column
        if label == "1": 
            label = np.ones((image_size * image_size,), dtype=int)  # ****** Labeling all the not effected image with O ************************
        elif label == "0":
            label = np.zeros((image_size * image_size,), dtype=int)  # ****** Labeling all the not effected image with O ************************   
        # now add the labels
        df["label"] = label
        
        ######################################
        # Update dataframe for images to include details for each image in the loop
        dataset = dataset._append(df)

    return dataset

effected_dataset = read_image(effected_img_path, label=1)
print("Shape of effected Image features", effected_dataset.shape)
not_effected_dataset = read_image(not_effected_img_path, label=0)
print("Shape of not effected Image features", not_effected_dataset.shape)


################################################################
#  STEP 3: GET DATA READY FOR RANDOM FOREST (or other classifier)
# COMBINE BOTH DATAFRAMES INTO A SINGLE DATASET
###############################################################
dataset = pd.concat([effected_dataset, not_effected_dataset], axis=0)  # Concatenate both image and mask datasets
print(dataset.head(5))
print('Shape of FULL dataset', dataset.shape)

# Assign training features to X and labels to Y
# Drop columns that are not relevant for training (non-features)
X = dataset.drop(labels=["label"], axis=1)

# Assign label values to Y (our prediction)
Y = dataset["label"].values

##Split data into train and test to verify accuracy after fitting the model.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

print('Shape of X_Train',X_train.shape)

####################################################################
# STEP 4: Define the classifier and fit a model with our training data
###################################################################

#Import training classifier
from sklearn.ensemble import RandomForestClassifier
## Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 42, verbose=2, n_jobs= 24)

## Train the model on training data
model.fit(X_train, y_train)

#######################################################
# STEP 5: Accuracy check
#########################################################

from sklearn import metrics
prediction_test = model.predict(X_test)
##Check accuracy on test dataset. 
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

#One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

##########################################################
#STEP 6: SAVE MODEL FOR FUTURE USE
###########################################################
##You can store the model for future use. In fact, this is how you do machine elarning
##Train on training images, validate on test images and deploy the model on unknown images. 
#
#
##Save the trained model as pickle string to disk for future use
from pathlib import Path
root = Path(".")
model_name = root/"hand_crafted_features_RF_model"
pickle.dump(model, open(model_name, 'wb')) # Here 'wb' stands for write binary
#To test the model on future datasets
#loaded_model = pickle.load(open(model_name, 'rb')) # Here 'rb' stands for read the binary


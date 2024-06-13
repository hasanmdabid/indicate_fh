# pylint: disable-all
import logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
from IPython.display import Image, display
from keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import random
from skimage.transform import resize
from IPython.display import Image as imgdisp, display
from keras.models import load_model
from clear_folder import clear_folder
import keras

# Define the paths to the folders and the weights file
effected_folder = '/home/abidhasan/Documents/Indicate_FH/data/effected'
not_effected_folder = '/home/abidhasan/Documents/Indicate_FH/data/not_effected'
#weights_path = '/home/abidhasan/Documents/Indicate_FH/saved_model/vgg16_300.h5'

models = ['xception', 'mobilenetv2', 'vgg16', 'inceptionv3']

n_channels = 3

# Function to preprocess images
img1 = "/home/abidhasan/Documents/Indicate_FH/train_val_test/test/effected/001_CLE204_Baseline_111.png"
img2 = "/home/abidhasan/Documents/Indicate_FH/data/effected/001_CLE114_Baseline_112.png"
img3 = "/home/abidhasan/Documents/Indicate_FH/data/effected/007_CLE203_Weizen_114.png"
img4 = "/home/abidhasan/Documents/Indicate_FH/data/effected/CLE101_Baseline_Reaktion_CLE101_Baseline_Reaktion_3.png"
img5 = "/home/abidhasan/Documents/Indicate_FH/data/effected/003_CLE170_Soja_78.png"
img6 = "/home/abidhasan/Documents/Indicate_FH/data/effected/004_CLE191_Weizen_58.png"
img7 = "/home/abidhasan/Documents/Indicate_FH/data/effected/004_CLE226_Soja_107.png"
img8 = "/home/abidhasan/Documents/Indicate_FH/data/effected/003_CLE219_Milch_165.png"
img9 = "/home/abidhasan/Documents/Indicate_FH/data/effected/003_CLE151_Trockenhefe_31.png"
img10 = "/home/abidhasan/Documents/Indicate_FH/data/effected/003_CLE170_Soja_104.png"
img11 = "/home/abidhasan/Documents/Indicate_FH/data/not_effected/001_CLE106_Baseline_35.png"
img12 = "/home/abidhasan/Documents/Indicate_FH/data/not_effected/CLE131_Soja_80.png"
img13 = "/home/abidhasan/Documents/Indicate_FH/data/not_effected/010_CLE185_Trockenhefe_72.png"
img14 = "/home/abidhasan/Documents/Indicate_FH/data/not_effected/009_CLE185_Weizenmehl_7.png"
img15 = "/home/abidhasan/Documents/Indicate_FH/data/not_effected/006_CLE220_Hefe_16.png"

destination_dir = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/'


#Select the set of images to be used

images = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15]



def get_img_array(img_path, size, image_nr):
    # `img` is a PIL image of size sizexsize
    img = keras.utils.load_img(img_path, target_size=size)
    img.save(destination_dir+'original_'+f"{image_nr}"+'.png')
    # `array` is a float32 Numpy array of shape (size, size, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, size, size, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


for model_name in models:
    print(model_name)
    if model_name == 'xception':
        size = 299
        last_conv_layer_name = "block14_sepconv2"
        
    elif model_name == 'mobilenetv2':
        size = 224
        last_conv_layer_name = "block_16_depthwise"
        
    elif model_name == 'vgg16':
        size = 224
        last_conv_layer_name = 'block5_conv3'
        
    elif model_name == 'inceptionv3':
        size = 299
        last_conv_layer_name = 'mixed10'
    
    model = load_model('/home/abidhasan/Documents/Indicate_FH/saved_model/'+model_name+'_100.h5')


    # Remove last layer's softmax
    model.layers[-1].activation = None

    
    for img_idx, imagepath in enumerate(images):
        img_array = get_img_array(imagepath, size=(size, size), image_nr=img_idx)
        # Creat teh heatmap image
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        
        def save_and_display_gradcam(imagepath, heatmap, cam_path='/home/abidhasan/Documents/Indicate_FH/grad_cam_images/'+model_name+'_'+ f"{img_idx}" +'.png', alpha=0.4):
            # Load the original image
            img = keras.utils.load_img(imagepath)
            img = keras.utils.img_to_array(img)
            print("Image shape {}".format(img.shape))
            print("Image Max Pixel value: {}".format(np.max(img)))

            # Rescale heatmap to a range 0-255
            heatmap = np.uint8(255 * heatmap)

            # Use jet colormap to colorize heatmap
            jet = mpl.colormaps["jet"]

            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]

            # Create an image with RGB colorized heatmap
            jet_heatmap = keras.utils.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
            jet_heatmap = keras.utils.img_to_array(jet_heatmap)

            print('Shape of Jet_heatmap: {}'.format(jet_heatmap.shape))
            print("Jet Heatmap Max Pixel value: {}".format(np.max(jet_heatmap)))

            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * alpha + img
            superimposed_img = keras.utils.array_to_img(superimposed_img)

            # Save the superimposed image
            superimposed_img.save(cam_path)

        save_and_display_gradcam(imagepath, heatmap)



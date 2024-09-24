# pylint: disable-all
import cv2
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import load_img
from gradcamutils import read_and_preprocess_img, GradCam, GradCamPlusPlus, ScoreCam, superimpose

# Please enter the directory address to save the score cam imposed images 
filtered_images_directory = '/home/abidhasan/Documents/Indicate_FH/score_cam_images'


# Lis the images path that you want to display the score cam effect. 
img1 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_0.png'
img2 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_1.png'
img3 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_2.png'
img4 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_3.png'
img5 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_4.png'
img6 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_5.png'
img7 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_6.png'
img8 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_7.png'
img9 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_8.png'
img10 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_9.png'
img11 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_10.png'
img12 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_11.png'
img13 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_12.png'
img14 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_13.png'
img15 = '/home/abidhasan/Documents/Indicate_FH/grad_cam_images/original_14.png'

model_names = ['xception', 'mobilenetv2', 'vgg16', 'inceptionv3']

"""

# Image paths
image_paths = [img1, img2, img3, img4, img5]

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))

# Set the top row titles
axes[0, 0].set_title('Original', fontsize=15)
for j, model_name in enumerate(model_names):
    axes[0, j + 1].set_title(model_name, fontsize=15)

# Plot the images and Score-CAM visualizations
for i, img_path in enumerate(image_paths):
    orig_img = np.array(load_img(img_path), dtype=np.uint8)
    axes[i, 0].imshow(orig_img)
    axes[i, 0].axis('off')

    for j, model_name in enumerate(model_names):
        model, img_array, layer_name = read_and_preprocess_img(
            img_path, model_name)
        score_cam = ScoreCam(model, img_array, layer_name)
        score_cam_superimposed = superimpose(img_path, score_cam)

        ax = axes[i, j + 1]
        ax.imshow(score_cam_superimposed)
        ax.axis('off')

plt.savefig(filtered_images_directory+'/score_cam_output_1.eps',
            format='eps', bbox_inches='tight')


# Image paths
image_paths = [img6, img7, img8, img9, img10]

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))

# Set the top row titles
axes[0, 0].set_title('Original', fontsize=15)
for j, model_name in enumerate(model_names):
    axes[0, j + 1].set_title(model_name, fontsize=15)

# Plot the images and Score-CAM visualizations
for i, img_path in enumerate(image_paths):
    orig_img = np.array(load_img(img_path), dtype=np.uint8)
    axes[i, 0].imshow(orig_img)
    axes[i, 0].axis('off')

    for j, model_name in enumerate(model_names):
        model, img_array, layer_name = read_and_preprocess_img(
            img_path, model_name)
        score_cam = ScoreCam(model, img_array, layer_name)
        score_cam_superimposed = superimpose(img_path, score_cam)

        ax = axes[i, j + 1]
        ax.imshow(score_cam_superimposed)
        ax.axis('off')

plt.savefig(filtered_images_directory+'/score_cam_output_2.eps',
            format='eps', bbox_inches='tight')

"""
# Image paths
image_paths = [img11, img12, img13, img14, img15]

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))

# Set the top row titles
axes[0, 0].set_title('Original', fontsize=15)
for j, model_name in enumerate(model_names):
    axes[0, j + 1].set_title(model_name, fontsize=15)

# Plot the images and Score-CAM visualizations
for i, img_path in enumerate(image_paths):
    orig_img = np.array(load_img(img_path), dtype=np.uint8)
    axes[i, 0].imshow(orig_img)
    axes[i, 0].axis('off')

    for j, model_name in enumerate(model_names):
        model, img_array, layer_name = read_and_preprocess_img(
            img_path, model_name)
        score_cam = ScoreCam(model, img_array, layer_name)
        score_cam_superimposed = superimpose(img_path, score_cam)

        ax = axes[i, j + 1]
        ax.imshow(score_cam_superimposed)
        ax.axis('off')

plt.savefig(filtered_images_directory+'/score_cam_output_3.eps',
            format='eps', bbox_inches='tight')

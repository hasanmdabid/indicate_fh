def image_generator(SIZE, batch_size, percentage_to_read, effected_aug_data_folder, not_effected_aug_data_folder, effected_image_directory, not_effected_image_directory):
        
    """
    @author: Md Abid HasaN
    Image shifts via the width_shift_range and height_shift_range arguments.
    Image flips via the horizontal_flip and vertical_flip arguments.
    Image rotations via the rotation_range argument
    Image brightness via the brightness_range argument.
    Image zoom via the zoom_range argument.
    """
    from keras.preprocessing.image import ImageDataGenerator
    import random
    import numpy as np
    from skimage import io
    import os
    from PIL import Image
    from clear_folder import clear_folder

    SIZE = SIZE
    batch_size = batch_size
    percentage_to_read = percentage_to_read  # Change this value as needed
    ################################################################
    #Clear the folder where you want to save the augmented images 
    # Specify the folder path you want to clear
    effected_aug_data_folder = effected_aug_data_folder
    not_effected_aug_data_folder = not_effected_aug_data_folder

    # Specify the folder path you want to read the images from
    effected_image_directory = effected_image_directory+'/'
    not_effected_image_directory = not_effected_image_directory+'/'

    # Call the clear_folder function to clear the contents of the folder
    clear_folder(effected_aug_data_folder)
    clear_folder(not_effected_aug_data_folder)

    # Construct an instance of the ImageDataGenerator class
    # Pass the augmentation parameters through the constructor. 

    datagen = ImageDataGenerator(
            rotation_range=90,     #Random rotation between 0 and 45
            width_shift_range=0.2,   #% shift
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='constant')    #Also try nearest, constant, reflect, wrap

    ####################################################################
    #Multiple images.
    #Manually read each image and create an array to be supplied to datagen via flow method


    def read_images_to_array(directory_path, target_size=(SIZE, SIZE)):
        image_list = []

        for filename in os.listdir(directory_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more image formats as needed
                img = Image.open(os.path.join(directory_path, filename)).convert('RGB')
                img = img.resize(target_size)  # Resize the image to the target size
                img_array = np.array(img)
                
                # Check if the image is grayscale and expand dimensions if necessary
                if len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, axis=2)
                
                image_list.append(img_array)

        images_as_array = np.array(image_list)

        return images_as_array


    # Directory path containing the images

    x = read_images_to_array(effected_image_directory)

    # Now, you have all the images as a 4D NumPy array in 'x'
    # The shape of the array will be (num_images, 256, 256, 3) for RGB images
    # If your images are grayscale, the shape will be (num_images, 256, 256, 1)
    print(x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=batch_size,  
                            save_to_dir=effected_aug_data_folder, 
                            save_prefix='aug', 
                            save_format='png'):
        i += 1
        if i > 250:
            break  # otherwise the generator would loop indefinitely  
        


    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(not_effected_image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Calculate the number of images to read based on the percentage
    num_images_to_read = int(len(image_files) * (percentage_to_read / 100))

    # Randomly shuffle the list of image files
    random.shuffle(image_files)

    # Initialize an empty list to store the images
    not_effected_images = []

    # Loop through the selected number of images and read them
    for i in range(num_images_to_read):
        image_file = image_files[i]
        image_path = os.path.join(not_effected_image_directory, image_file)
        
        # Read the image using SKIMAGE
        image = io.imread(image_path)

        if image is not None:
            # Append the image to the list
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE,SIZE))
            not_effected_images.append(np.array(image))
            #not_effected_images.append(image)
        else:
            print(f"Failed to read {image_file}")

    # Convert the list of images into a NumPy array
    x = np.array(not_effected_images)

    # Print the shape of the resulting NumPy array
    print("Shape of 30% randomly selected not effected data:", x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=batch_size,  
                            save_to_dir=not_effected_aug_data_folder, 
                            save_prefix='aug', 
                            save_format='png'):
        i += 1
        if i > x.shape[0]//batch_size:
            break  # otherwise the generator would loop indefinitely  
            
            
            # Delete variables
    import gc
    del x
    # Run garbage collection to clear memory
    gc.collect()
    
    

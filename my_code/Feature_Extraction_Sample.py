"""
This script will generate handcrafted features form the CLE images
Script Author: Md Abid Hasan
Project: Indicate_FH
Date: 12 July 2023

"""

from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from scipy import ndimage as nd
from skimage import filters
import pandas as pd
from skimage.transform import resize  

############################ Denoising filters ###############

# Img_as_flot to single-precision (32-bit) floating point format, with values in [0, 1].
img_ne = img_as_float(io.imread("/home/abidhasan/Documents/Indicate_fh/image/noteffected2.jpeg", as_gray=True))
img_e = img_as_float(io.imread("/home/abidhasan/Documents/Indicate_fh/image/effected.png", as_gray=True))
#Need to convert to float as we will be doing math on the array
#To read a pack of imahes we have to use io.imread_collection(load_pattern, conserve_memory=True)

# Applying the Contrast limited adaptive histogram equalization to increase the contrast of the image
img_CLAHE_ne = equalize_adapthist(img_ne, kernel_size=None, clip_limit=0.1, nbins=256)
img_CLAHE_e = equalize_adapthist(img_e, kernel_size=None, clip_limit=0.1, nbins=256)

plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/image_ne_CLAHE.jpg", img_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/image_e_CLAHE.jpg", img_e)


# Appling the Gaussian filter with a Gaussian filter Scipy 
gaussian_ne = nd.gaussian_filter(img_ne, sigma=3)
gaussian_e = nd.gaussian_filter(img_e, sigma=3)
gaussian_CLAHE_ne = nd.gaussian_filter(img_CLAHE_ne, sigma=3)
gaussian_CLAHE_e = nd.gaussian_filter(img_CLAHE_e, sigma=3)

plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/gaussian_ne.jpg", gaussian_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/gaussian_e.jpg", gaussian_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/gaussian_CLAHE_ne.jpg", gaussian_CLAHE_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/gaussian_CLAHE_e.jpg", gaussian_CLAHE_e)



# Appling the Mean filter with Scipy-----------------------------------------------------------------------------
median_ne = nd.median_filter(img_ne, size=3)
median_e = nd.median_filter(img_e, size=3)
median_CLAHE_ne = nd.median_filter(img_CLAHE_ne, size=3)
median_CLAHE_e = nd.median_filter(img_CLAHE_e, size=3)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/median_e.jpg", median_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/median_ne.jpg", median_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/median_CLAHE_e.jpg", median_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/median_CLAHE_ne.jpg", median_CLAHE_ne)

# Appling the PREWITT filter with 
prewitt_ne = filters.prewitt(img_ne)
prewitt_e = filters.prewitt(img_e)
prewitt_CLAHE_ne = filters.prewitt(img_CLAHE_ne)
prewitt_CLAHE_e = filters.prewitt(img_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/prewitt_e.jpg", prewitt_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/prewitt_ne.jpg", prewitt_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/prewitt_CLAHE_e.jpg", prewitt_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/prewitt_CLAHE_ne.jpg", prewitt_CLAHE_ne)

## Appling the Sobel filter with Scipy-----------------------------------------------------------------------------
sobel_ne = filters.sobel(img_ne)
sobel_e = filters.sobel(img_e)
sobel_CLAHE_ne = filters.sobel(img_CLAHE_ne)
sobel_CLAHE_e = filters.sobel(img_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/sobel_e.jpg", sobel_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/sobel_ne.jpg", sobel_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/sobel_CLAHE_e.jpg", sobel_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/sobel_CLAHE_ne.jpg", sobel_CLAHE_ne)

dog_ne = filters.difference_of_gaussians(img_ne, low_sigma= 5, high_sigma=None)
dog_e  = filters.difference_of_gaussians(img_e, low_sigma= 5, high_sigma=None)
dog_CLAHE_e  = filters.difference_of_gaussians(img_CLAHE_e, low_sigma= 5, high_sigma=None)
dog_CLAHE_ne  = filters.difference_of_gaussians(img_CLAHE_ne, low_sigma= 5, high_sigma=None)

plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/dog_e.jpg", dog_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/dog_ne.jpg", dog_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/dog_CLAHE_e.jpg", dog_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/dog_CLAHE_ne.jpg", dog_CLAHE_ne)

farid_ne = filters.farid(img_ne, mode='constant')
farid_e  = filters.farid(img_e, mode='constant')
farid_CLAHE_e  = filters.farid(img_CLAHE_e, mode='constant')
farid_CLAHE_ne  = filters.farid(img_CLAHE_ne, mode='constant')

plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/farid_e.jpg", farid_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/farid_ne.jpg", farid_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/farid_CLAHE_e.jpg", farid_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/farid_CLAHE_ne.jpg", farid_CLAHE_ne)


butter_ne = filters.butterworth(img_ne, cutoff_frequency_ratio=0.005, high_pass=True, order=7.0, squared_butterworth=True, npad=0)
butter_e = filters.butterworth(img_e, cutoff_frequency_ratio=0.005, high_pass=True, order=7.0, squared_butterworth=True, npad=0)
butter_CLAHE_e = filters.butterworth(img_CLAHE_e, cutoff_frequency_ratio=0.005, high_pass=True, order=7.0, squared_butterworth=True, npad=0)
butter_CLAHE_ne = filters.butterworth(img_CLAHE_ne, cutoff_frequency_ratio=0.005, high_pass=True, order=7.0, squared_butterworth=True, npad=0)

plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/Butter_e.jpg", butter_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/Butter_ne.jpg", butter_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/Butter_CLAHE_e.jpg", butter_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/Butter_CLAHE_ne.jpg", butter_CLAHE_ne)

robert_ne = filters.roberts(img_ne, mask=None)
robert_e  = filters.roberts(img_e, mask=None)
robert_CLAHE_e  = filters.roberts(img_CLAHE_e, mask=None)
robert_CLAHE_ne  = filters.roberts(img_CLAHE_ne, mask=None)

plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/robert_e.jpg", robert_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/robert_ne.jpg", robert_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/robert_CLAHE_e.jpg", robert_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/robert_CLAHE_ne.jpg", robert_CLAHE_ne)

from skimage.filters.rank import entropy
from skimage.morphology import disk
import pandas as pd

entropy_e = entropy(img_as_ubyte(img_e), disk(1))
entropy_ne = entropy(img_as_ubyte(img_ne), disk(1))
entropy_CLAHE_e = entropy(img_as_ubyte(img_CLAHE_e), disk(1))
entropy_CLAHE_ne = entropy(img_as_ubyte(img_CLAHE_ne), disk(1))

plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/entropy_e.jpg", entropy_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/entropy_ne.jpg", entropy_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/entropy_CLAHE_e.jpg", entropy_CLAHE_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/entropy_CLAHE_ne.jpg", entropy_CLAHE_ne)

# Applying Non Local Median filter--------------------------------------------------------------------------------

sigma_est = np.mean(estimate_sigma(img_CLAHE_ne))
sigma_est = np.mean(estimate_sigma(img_CLAHE_e))

nlm_ne = denoise_nl_means(img_ne, h=1.7 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)
nlm_e = denoise_nl_means(img_e, h=1.7 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)


nlm_CLAHE_ne = denoise_nl_means(img_CLAHE_ne, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)
nlm_CLAHE_e = denoise_nl_means(img_CLAHE_e, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)

print('Maximume value inside nlm_CLAHE_ne image array:', np.max(nlm_CLAHE_ne))
print('Minimume value inside nlm_CLAHE_ne image array:', np.min(nlm_CLAHE_e))


plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/NLM_ne.jpg",nlm_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/NLM_e.jpg",nlm_e)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/NLM_CLAHE_ne.jpg",nlm_CLAHE_ne)
plt.imsave("/home/abidhasan/Documents/Indicate_fh/image/NLM_CLAHE_e.jpg",nlm_CLAHE_e)


# Resizing the image---------------------------------------------------------------------------------------------------


img_e = img_as_ubyte(resize(img_e, (256,256),anti_aliasing=True))
img_ne = img_as_ubyte(resize(img_ne, (256,256),anti_aliasing=True))

img_CLAHE_e = img_as_ubyte(resize(img_CLAHE_e, (256,256), anti_aliasing=True))
img_CLAHE_en = img_as_ubyte(resize(img_CLAHE_ne, (256,256), anti_aliasing=True))

gaussian_e = img_as_ubyte(resize(gaussian_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
gaussian_ne = img_as_ubyte(resize(gaussian_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
gaussian_CLAHE_e = img_as_ubyte(resize(gaussian_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
gaussian_CLAHE_ne = img_as_ubyte(resize(gaussian_CLAHE_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer

prewitt_e = img_as_ubyte(resize(prewitt_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
prewitt_ne = img_as_ubyte(resize(prewitt_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
prewitt_CLAHE_e = img_as_ubyte(resize(prewitt_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
prewitt_CLAHE_ne = img_as_ubyte(resize(prewitt_CLAHE_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer

sobel_e = img_as_ubyte(resize(sobel_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
sobel_ne = img_as_ubyte(resize(sobel_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
sobel_CLAHE_e = img_as_ubyte(resize(sobel_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
sobel_CLAHE_ne = img_as_ubyte(resize(sobel_CLAHE_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
    
median_e = img_as_ubyte(resize(median_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
median_ne = img_as_ubyte(resize(median_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
median_CLAHE_e = img_as_ubyte(resize(median_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
median_CLAHE_ne = img_as_ubyte(resize(median_CLAHE_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer

nlm_e = img_as_ubyte(resize(nlm_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
nlm_ne = img_as_ubyte(resize(nlm_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
nlm_CLAHE_e = img_as_ubyte(resize(nlm_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
nlm_CLAHE_ne = img_as_ubyte(resize(nlm_CLAHE_ne, (256, 256), anti_aliasing=True)) 

farid_e = img_as_ubyte(resize(farid_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
farid_ne = img_as_ubyte(resize(farid_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
farid_CLAHE_e = img_as_ubyte(resize(farid_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
farid_CLAHE_ne = img_as_ubyte(resize(farid_CLAHE_ne, (256, 256), anti_aliasing=True)) 

dog_e = img_as_ubyte(resize(dog_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
dog_ne = img_as_ubyte(resize(dog_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
dog_CLAHE_e = img_as_ubyte(resize(dog_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
dog_CLAHE_ne = img_as_ubyte(resize(dog_CLAHE_ne, (256, 256), anti_aliasing=True)) 

butter_e = img_as_ubyte(resize(butter_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
butter_ne = img_as_ubyte(resize(butter_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
butter_CLAHE_e = img_as_ubyte(resize(butter_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
butter_CLAHE_ne = img_as_ubyte(resize(butter_CLAHE_ne, (256, 256), anti_aliasing=True)) 
 
robert_e = img_as_ubyte(resize(robert_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
robert_ne = img_as_ubyte(resize(robert_ne, (256, 256), anti_aliasing=True))             #Check dimensions in variable explorer
robert_CLAHE_e = img_as_ubyte(resize(robert_CLAHE_e, (256, 256), anti_aliasing=True))               #Check dimensions in variable explorer 
robert_CLAHE_ne = img_as_ubyte(resize(robert_CLAHE_ne, (256, 256), anti_aliasing=True)) 

print(np.max(robert_ne))     
      
entropy_e = resize(entropy_e, (256, 256), anti_aliasing=True)               #Check dimensions in variable explorer 
entropy_ne = resize(entropy_ne, (256, 256), anti_aliasing=True)              #Check dimensions in variable explorer
entropy_CLAHE_e = resize(entropy_CLAHE_e, (256, 256), anti_aliasing=True)              #Check dimensions in variable explorer 
entropy_CLAHE_ne = resize(entropy_CLAHE_ne, (256, 256), anti_aliasing=True)

img_e = img_e.reshape(-1)
img_ne = img_ne.reshape(-1)
img_CLAHE_en = img_CLAHE_en.reshape(-1)
img_CLAHE_e = img_CLAHE_e.reshape(-1)

gaussian_e = gaussian_e.reshape(-1)
gaussian_ne = gaussian_ne.reshape(-1)
gaussian_CLAHE_e = gaussian_CLAHE_e.reshape(-1)
gaussian_CLAHE_ne = gaussian_CLAHE_ne.reshape(-1)

median_e = median_e.reshape(-1)
median_ne  =median_ne.reshape(-1)
median_CLAHE_e = median_CLAHE_e.reshape(-1)
median_CLAHE_ne = median_CLAHE_ne.reshape(-1)

prewitt_e = prewitt_e.reshape(-1)
prewitt_ne  =prewitt_ne.reshape(-1)
prewitt_CLAHE_e = prewitt_CLAHE_e.reshape(-1)
prewitt_CLAHE_ne = prewitt_CLAHE_ne.reshape(-1)

sobel_e = sobel_e.reshape(-1)
sobel_ne  =sobel_ne.reshape(-1)
sobel_CLAHE_e = sobel_CLAHE_e.reshape(-1)
sobel_CLAHE_ne = sobel_CLAHE_ne.reshape(-1)

dog_e = dog_e.reshape(-1)
dog_ne  =dog_ne.reshape(-1)
dog_CLAHE_e = dog_CLAHE_e.reshape(-1)
dog_CLAHE_ne = dog_CLAHE_ne.reshape(-1)

farid_e = farid_e.reshape(-1)
farid_ne  =farid_ne.reshape(-1)
farid_CLAHE_e = farid_CLAHE_e.reshape(-1)
farid_CLAHE_ne = farid_CLAHE_ne.reshape(-1)

butter_e = butter_e.reshape(-1)
butter_ne  =butter_ne.reshape(-1)
butter_CLAHE_e = butter_CLAHE_e.reshape(-1)
butter_CLAHE_ne = butter_CLAHE_ne.reshape(-1)

robert_e = robert_e.reshape(-1)
robert_ne  =robert_ne.reshape(-1)
robert_CLAHE_e = robert_CLAHE_e.reshape(-1)
robert_CLAHE_ne = robert_CLAHE_ne.reshape(-1)

print(np.max(robert_ne))

entropy_e = entropy_e.reshape(-1)
entropy_ne  =entropy_ne.reshape(-1)
entropy_CLAHE_e = entropy_CLAHE_e.reshape(-1)
entropy_CLAHE_ne = entropy_CLAHE_ne.reshape(-1)

df_ne = pd.DataFrame()
df_ne['image'] = img_ne
df_ne['image_CLAHE'] = img_CLAHE_en
df_ne['gaussian'] = gaussian_ne
df_ne['gaussian_CLAHE'] = gaussian_CLAHE_ne 
df_ne['median'] = median_ne
df_ne['median_CLAHE'] = median_CLAHE_ne
df_ne['prewitt'] = prewitt_e
df_ne['prewitt_CLAHE'] = prewitt_CLAHE_ne            
df_ne['sobel'] = sobel_ne
df_ne['sobel_CLAHE'] = sobel_CLAHE_ne
df_ne['dog'] = dog_ne
df_ne['dog_CLAHE'] = dog_CLAHE_ne
df_ne['farid'] = farid_ne
df_ne['farid_CLAHE'] = farid_CLAHE_ne
df_ne['butter'] = butter_ne
df_ne['butter_CLAHE'] = butter_CLAHE_ne
df_ne['robert'] = robert_ne
df_ne['robert_CLAHE']  = robert_CLAHE_ne
df_ne['entropy'] = entropy_ne
df_ne['entropy_CLAHE'] = entropy_CLAHE_ne           

print(df_ne.shape)
print(np.max(df_ne.loc[:,"robert"]))

data= df_ne.to_numpy()
print(np.max(data))





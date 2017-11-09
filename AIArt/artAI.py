import numpy as np
from keras.applications import vgg16
from keras import backend as K 
from keras.preprocessing.image import load_img, img_to_array
#Helper Class
import image_processing as img 
#Image Display
from IPython.display import Image

base_image = K.variable(img.preprocess_image('./base_image.png'))
style_reference_image = K.variable(img.preprocess_image('./style_image.png'))
combination_image = K.placeholder((1,400,711,3))

Image('./style_image.png')

#Combine 3 Images Into A Single Keras Tensor
input_tensor = K.concatenate((base_image, style_reference_image, combination_image, axis == 0))

#Build The VGG16 Network With Our 3 Images As Input
model = vgg16.VGG16(input_tensor=input_tensor, weights = 'imagenet', include_top = False)
print('Model Loaded!')

#Combine Loss Functions Into A Singular Scalar
loss = img.combination_loss(model, combination_image)
print(loss)

Tensor("add_16:0", shape=(), dtype=float32)
#Get The Gradients Of The Generated Image Wrt The Loss
grads = K.gradients(loss, combination_image)
print(grads)

#Run Optimization (L-BFGS) Over The Pixels Of The Generated Image
#To Minimize The Loss
combination_image = img.minimize_loss(grads, loss, combination_image)
Image(combination_image)
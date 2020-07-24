from tensorflow import keras
import numpy as np
from keras.preprocessing import image

# load model

model = keras.models.load_model("classifier")

def pred(file):
    #print(f'image: {file}')
    test_image = image.load_img(file, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    #training_set.class_indices
    return result
    #if result[0][0] == 1:
    #  return 'dog'
    #else:
    #  return 'cat'
    
for i in range(50000):
    pred('dataset/test_set/cats/cat.4411.jpg') 
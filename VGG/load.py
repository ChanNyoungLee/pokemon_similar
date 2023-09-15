import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg16

class DataLoad:
    
    def __init__(self):
        pass

    def load(self):

        feature_df = pd.DataFrame()
        pokemon_name = []
        im = []
        im_array = []

        model = vgg16
        base_model = model.VGG16(weights='imagenet')
        vgg_16 = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)


        for num in range(len(os.listdir('./pokemon/images/images/'))):
            name = os.listdir('./pokemon/images/images/')[num]
            pokemon_name.append(os.listdir('./pokemon/images/images/')[num].split('.')[0])
            image = Image.open(f'./pokemon/images/images/{name}').convert('RGB')
            im.append(image)

            image = image.resize((224,224))

            image = image.convert('RGB')


            im_array.append(np.array(image))

            pre = np.array(im_array)

        feature = vgg_16.predict(pre)

        df = pd.DataFrame(feature)


        feature_df = pd.concat([feature_df,df],axis=0)

        print('_data_load')
        return pokemon_name, im, feature_df  # Return both the names and images
    

    # def _resize(self, x):
    #     return x.resize((224, 224))

    # def _convert(self, x):
    #     return x.convert('RGB')

    # def _array(self, x):
    #     return img_to_array(x)

    # def _expand(self, x):
    #     return np.expand_dims(x, axis=0)

    # def _preprocess(self, x):
    #     return preprocess_input(x)

    # def load(self):
    #     pokemon_name, im = self._data_load()  # Unpack the returned values
    #     images = []
    #     for image in im:
    #         # image = self._resize(image)
    #         # image = self._convert(image)
    #         # image = self._array(image)
    #         # image = self._expand(image)
    #         # image = self._preprocess(image)
    #         # images.append(image)

    #     print('load')

    #     return pokemon_name, images , im
    
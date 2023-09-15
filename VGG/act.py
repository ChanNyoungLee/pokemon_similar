from load import DataLoad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from IPython.display import display

class act:
    

    def __init__(self, name):
        data_loader = DataLoad()
        pokemon_names, im, feature_df = data_loader.load()

        self.pokemon_name = pokemon_names
        self.im = im
        self.feature_df = feature_df
        self.name = name



    def _dataframe(self):
        image_df = pd.DataFrame(self.im)
    
        image_df['name'] = self.pokemon_name
        return image_df

        

    # def _transpose(self):
    #     self.image_df = self.image_df.T
    
    def _cosine(self):

        _feature_df = pd.DataFrame(self.feature_df)
        _feature_df.index = self.pokemon_name
    
   
        feature_cosine = cosine_similarity(_feature_df)
        feature_cosine = pd.DataFrame(feature_cosine)
        feature_cosine.columns=self.pokemon_name
        feature_cosine.index = self.pokemon_name

        return feature_cosine
    
    # def _show(self):
    #     fig = plt.figure()
    #     rows = 2
    #     cols = 3
    #     name = self.name

    def show(self):
        
        fig = plt.figure()
        rows = 2
        cols = 3

        image_df = self._dataframe()


        feature_cosine = self._cosine()
        feature_cosine = feature_cosine.mul(100).round(2).astype('str') + '%'

        # name = self.name


        # main
        
        main_image = image_df[image_df['name'] == self.name].iloc[0,0]
        ax1 = fig.add_subplot(rows,cols,1)
        ax1.imshow(main_image)
        ax1.set_title(self.name)
        ax1.axis('off')

   

        # first
        first_name = feature_cosine[self.name].sort_values(ascending = False)[1:6].index[0]
        first_value = feature_cosine[self.name].sort_values(ascending = False)[1:6][0]
        first_image = image_df[image_df['name'] == first_name].iloc[0,0]
        first = first_name + ' ' + str(first_value)

        ax2 = fig.add_subplot(rows,cols,2)
        ax2.imshow(first_image)
        ax2.set_title(first)
        ax2.axis('off')

        # second
        second_name = feature_cosine[self.name].sort_values(ascending = False)[1:6].index[1]
        second_value = feature_cosine[self.name].sort_values(ascending = False)[1:6][1]
        second_image = image_df[image_df['name'] == second_name].iloc[0,0]
        second = second_name + ' ' + str(second_value)

        ax3 = fig.add_subplot(rows,cols,3)
        ax3.imshow(second_image)
        ax3.set_title(second)
        ax3.axis('off')


        # third
        third_name = feature_cosine[self.name].sort_values(ascending = False)[1:6].index[2]
        third_value = feature_cosine[self.name].sort_values(ascending = False)[1:6][2]
        third_image = image_df[image_df['name'] == third_name].iloc[0,0]
        third = third_name + ' ' + str(third_value)

        ax4 = fig.add_subplot(rows,cols,4)
        ax4.imshow(third_image)
        ax4.set_title(third)
        ax4.axis('off')



        # fourth
        fourth_name = feature_cosine[self.name].sort_values(ascending = False)[1:6].index[3]
        fourth_value = feature_cosine[self.name].sort_values(ascending = False)[1:6][3]
        fourth_image = image_df[image_df['name'] == fourth_name].iloc[0,0]
        fourth = fourth_name + ' ' + str(fourth_value)

        ax5 = fig.add_subplot(rows,cols,5)
        ax5.imshow(fourth_image)
        ax5.set_title(fourth)
        ax5.axis('off')


        
        # fifth
        fifth_name = feature_cosine[self.name].sort_values(ascending = False)[1:6].index[4]
        fifth_value = feature_cosine[self.name].sort_values(ascending = False)[1:6][4]
        fifth_image = image_df[image_df['name'] == fifth_name].iloc[0,0]
        fifth = fifth_name + ' ' + str(fifth_value)

        ax6 = fig.add_subplot(rows,cols,6)
        ax6.imshow(fifth_image)
        ax6.set_title(fifth)
        ax6.axis('off')

        print('이미지가 출력되었습니다.')


        plt.show()



    # def process(self):
    #     self._dataframe(self.im)
    #     self._rename()
    #     self._transpose()
    #     self._cosine()
    #     self._show()
    
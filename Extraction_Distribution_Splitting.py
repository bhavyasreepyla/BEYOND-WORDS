import os 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import splitfolders

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Input, concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix


#                                                           DATA EXTRACTION


# Path where our data is located
base_path = "C:/Users/USER/Desktop/BEYOND_WORDS/asl_alphabet_train/asl_alphabet_train/"

# Dictionary to save our 36 classes
categories = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "G", 10: "K", 11: "L", 12: "M",
              13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
              25: "Z", 26: "del", 27: "nothing", 28: "space"}

def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: base_path + x)
    return df

# list containing all the filenames in the dataset
filenames_list = []
# list to store the corresponding category, note that each folder of the dataset has one class of data
categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])
    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({"filename": filenames_list, "category": categories_list})
df = add_class_name_prefix(df, "filename")

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

print("number of elements = ", len(df))

#                                                            FINISHED


#                                                          EXPLORATION

plt.figure(figsize=(40, 40))

for i in range(100):
    category_folder = categories[df.category[i]]
    filename = os.path.basename(df.filename[i])
    path = os.path.join(base_path, category_folder, filename)
    img = plt.imread(path)
    plt.subplot(10, 10, i + 1)
    plt.imshow(img)
    plt.title(categories[df.category[i]], fontsize=35, fontstyle='italic')
    plt.axis("off")

plt.show()

#                                                           FINISHED

#                                                            DISTRIBUTION

label,count = np.unique(df.category,return_counts=True)
uni = pd.DataFrame(data=count,index=categories.values(),columns=['Count'])

plt.figure(figsize=(14,4),dpi=200)
sns.barplot(data=uni, x=uni.index, y='Count', hue=uni.index, palette='icefire', width=0.4, legend=False).set_title('Class distribution in Dataset', fontsize=15)
plt.show()

#                                                          FINISHED


#                                                         SPLITTING

import splitfolders

input_folder = 'C:/Users/USER/Desktop/BEYOND_WORDS/asl_alphabet_train/asl_alphabet_train/'
output_folder = 'C:/Users/USER/Desktop/BEYOND_WORDS/split_data/'

# Split the data
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None)  # 'group_prefix=None' for no grouping

print("done")


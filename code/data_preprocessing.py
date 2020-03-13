#importing necessary libraries
from tqdm import tqdm
from PIL import Image
import glob
import pandas as pd

#reading csv file which contains train image paths and labels
train_dataset = pd.read_csv("MURA-v1.1/train_labeled_studies.csv")

#choosing specific part of body
train_df = train_dataset.iloc[11438: , :]
train_df.columns = ['path', 'class']

#reading csv file which contains validation image paths and labels
validation_dataset = pd.read_csv("MURA-v1.1/valid_labeled_studies.csv")

#choosing specific part of body
validation_df = validation_dataset.iloc[369:536, :]
validation_df.columns = ['path', 'class']

#putting train images to their label folders to use flow_from_directory method of ImageDataGenerator Class
count = 0
for index in tqdm(train_df.index.values):
    folder_path = train_df['path'][index]
    if(train_df['class'][index] == 1):
        for img_file in glob.glob(folder_path + "/*.png"):
            img = Image.open(img_file)
            img_name = str(count)
            img.save('train_1/' + img_name + '.png')
            count = count + 1
    elif(train_df['class'][index] == 0):
        for img_file in glob.glob(folder_path + "/*.png"):
            img = Image.open(img_file)
            img_name = str(count)
            img.save('train_0/' + img_name + '.png')
            count = count + 1

#putting validation images to their label folders to use flow_from_directory method of ImageDataGenerator Class
count = 0
for index in tqdm(validation_df.index.values):
    folder_path = validation_df['path'][index]
    if(validation_df['class'][index] == 1):
        for img_file in glob.glob(folder_path + "/*.png"):
            img = Image.open(img_file)
            img_name = str(count)
            img.save('val_1/' + img_name + '.png')
            count = count + 1
    elif(validation_df['class'][index] == 0):
        for img_file in glob.glob(folder_path + "/*.png"):
            img = Image.open(img_file)
            img_name = str(count)
            img.save('val_0/' + img_name + '.png')
            count = count + 1
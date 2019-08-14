#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[28]:


import os
import re
import numpy as np
from PIL import Image
from shutil import copyfile


# In[22]:


raw_ds_path = '../dataset-raw/'


# ## Resizing images
# 
# Firstly, I'm going to resize images to size 150x150 pixels.

# In[30]:


width = 150
height = 150
resized_img_ds_path = '../dataset-resized'
# just to be sure that directory exists
if not os.path.exists(resized_img_ds_path):
    os.makedirs(resized_img_ds_path)

if os.path.isdir(raw_ds_path):
    all_files = os.listdir(raw_ds_path)
    for file in all_files:
        img = Image.open(os.path.join(raw_ds_path, file))
        img = img.resize((width, height), Image.ANTIALIAS)
        img.save(os.path.join(resized_img_ds_path, file))


# I'm going to make two lists containing paths to all dog files and all cat files.

# In[31]:


cat_files = list()
dog_files = list()

if os.path.isdir(resized_img_ds_path):
    all_files = os.listdir(resized_img_ds_path)
    for file in all_files:
        m = re.match("dog.+", file)
        if m is not None:
            dog_files.append(os.path.join(resized_img_ds_path, file))
        else:
            cat_files.append(os.path.join(resized_img_ds_path, file))


# I'm going to randomly split images into train and validation dataset. I don't need to make a test set because I don't set a random seed. What does it mean for me? It mean that I can split this dataset again after seting my hyperparameters and model's architecture and then train it again. It will allow me to estimate my test error.

# ## Spliting data
# I'm going to split dataset in stratified way.

# In[32]:


dog_ds_size = len(dog_files)
cat_ds_size = len(cat_files)

split_factor = 0.9 # 1.0 means all examples will be treated as training set

dog_indices = np.random.choice(dog_ds_size, int(split_factor*dog_ds_size), replace=False)
cat_indices = np.random.choice(cat_ds_size, int(split_factor*cat_ds_size), replace=False)


# ## Copying files
# Copying files into destination directory, splited to train and validation datasets. Make sure that your destination directiories are empty.

# In[33]:


dst_dir = 'dataset'
dst_train_dogs = os.path.join(dst_dir, 'train/dog')
dst_train_cats = os.path.join(dst_dir, 'train/cat')
dst_val_dogs = os.path.join(dst_dir, 'validation/dog')
dst_val_cats = os.path.join(dst_dir, 'validation/cat')
# just to be sure that directories exist
os.makedirs(dst_dir)
os.makedirs(dst_train_dogs)
os.makedirs(dst_train_cats)
os.makedirs(dst_val_dogs)
os.makedirs(dst_val_cats)

# copying dogs
for index in range(len(dog_files)):
    if index in dog_indices:
        copyfile(dog_files[index], os.path.join(dst_train_dogs, str(index)))
    else:
        copyfile(dog_files[index], os.path.join(dst_val_dogs, str(index)))
        
# copying cats
for index in range(len(cat_files)):
    if index in cat_indices:
        copyfile(cat_files[index], os.path.join(dst_train_cats, str(index)))
    else:
        copyfile(cat_files[index], os.path.join(dst_val_cats, str(index)))


# In[ ]:





import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2


import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task):

        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size,3))
        self.std = np.ones((hp.img_size,hp.img_size,3))
        self.calc_mean_and_std()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), task == '3', True, True)
                
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), task == '3', False, False)
        

    def calc_mean_and_std(self):

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img
        
        # Calculate the mean across the samples 
        self.mean = np.mean(data_sample, axis=0)

        # Calculate the standard deviation across the samples 
        self.std = np.std(data_sample, axis=0)

    def standardize(self, img):
        
        img = (img - self.mean) / self.std

        return img

    def preprocess_fn(self, img):
        img = img / 255.
        img = self.standardize(img)
        
        return img
    
        
    def get_data(self, path, is_vgg, shuffle, augment):
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)
            

        # VGG must take images of size 224x224
        img_size = 224

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)
        
        

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class
                    
            
        return data_gen
                
        

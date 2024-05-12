import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        
class Pretrained_model(tf.keras.Model):
    def __init__(self, model_name):
        super(Pretrained_model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()
        self.model_name = model_name

       # Initialize base models
        self.resnet = tf.keras.applications.ResNet50(
                input_shape=(224, 224, 3),
                weights='imagenet',
                classifier_activation='softmax',
                include_top=False)
        
        
        self.efficientnet = tf.keras.applications.EfficientNetV2L(
                input_shape=(224, 224, 3),
                weights='imagenet',
                classifier_activation='softmax',
                include_top=False)
        
        
        self.vgg = tf.keras.applications.VGG16(
                input_shape=(224, 224, 3),
                weights='imagenet',
                classifier_activation='softmax',
                include_top=False)
                
#         if (self.model_name == 'resnet'):
#             self.base_model = self.resnet
            
#         elif (self.model_name == 'efficientnet'):
#             self.base_model = self.efficientnet
            
#         elif (self.model_name == 'vgg'):
#             self.base_model = self.vgg
            
#         else:
#             print('Not specify')
#             return

        # For Base models
        self.head = [
              Flatten(),
              Dense(512, activation='relu'),
              Dense(128, activation='relu'),
              Dense(15, activation='softmax')
        ]

        # For fusion I
        self.head = [
            tf.keras.layers.Average()
        ]

        # For fusion II
        self.head = [
              Dense(15, activation='softmax')
        ]
        
        # For fusion III
        self.head = [
              Dense(128, activation='relu'),
              Dense(15, activation='softmax')
        ]

        
        
        
        self.head1 = [
              Flatten(),
              Dense(512, activation='relu', trainable=False),
              Dense(128, activation='relu', trainable=False),
              Dense(15, activation='softmax', trainable=False) 
        ]
        
        self.head2 = [
              Flatten(),
              Dense(512, activation='relu', trainable=False),
              Dense(128, activation='relu', trainable=False),
              Dense(15, activation='softmax', trainable=False) 
        ]
        
        self.head3 = [
              Flatten(),
              Dense(512, activation='relu', trainable=False),
              Dense(128, activation='relu', trainable=False),
              Dense(15, activation='softmax', trainable=False) 
        ]
        
#         self.head1 = [
#             Flatten(),
#               Dense(128, activation='relu')
#         ]
        
#         self.head2 = [
#             Flatten(),
#               Dense(128, activation='relu')
#         ]
        
#         self.head3 = [
#             Flatten(),
#               Dense(128, activation='relu')
#         ]
        
        
        # for layer in self.base_model.layers:
        #         layer.trainable = False
                
        for layer in self.vgg.layers:
                layer.trainable = False
                
        for layer in self.resnet.layers:
                layer.trainable = False
                
        for layer in self.efficientnet.layers:
                layer.trainable = False
        
        self.head = tf.keras.Sequential(self.head, name="model_head") 
        self.head1 = tf.keras.Sequential(self.head1, name="model_head1") 
        self.head2 = tf.keras.Sequential(self.head2, name="model_head2") 
        self.head3 = tf.keras.Sequential(self.head3, name="model_head3") 

    def call(self, x):
        """ Passes the image through the network. """
        
        x1 = self.vgg(x)
        x1 = self.head1(x1)
        
        x2 = self.resnet(x)
        x2 = self.head2(x2)
        
        x3 = self.efficientnet(x)
        x3 = self.head3(x3)
                
        # x = tf.keras.layers.Concatenate()([x1, x2, x3])
        x = self.head([x1, x2, x3])
        
        # x = self.base_model(x)
    
        # x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TASK 3
        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        #       Read the documentation carefully, some might not work with our 
        #       model!

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=False)


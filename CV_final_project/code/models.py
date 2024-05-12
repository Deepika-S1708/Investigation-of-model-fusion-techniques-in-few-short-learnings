import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        # Convolutional neural network with Dropout
        self.architecture = [
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(None, None, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(15, activation='softmax')
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TASK 1
        # TODO: Select a loss function for your network 
        #       (see the documentation for tf.keras.losses)

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TASK 3
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.Adam()

        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]


        # all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained.
        
        for layer in self.vgg16:
            layer.trainable = False

        # TODO: Write a classification head for our 15-scene classification task.

        self.head = [
              Flatten(),
              Dense(512, activation='relu'),
              Dense(128, activation='relu'),
              Dense(15, activation='softmax') 
        ]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TASK 3
        # TODO: Select a loss function for your network

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=False)
        
        
        
class Pretrained_model(tf.keras.Model):
    def __init__(self, model_name):
        super(Pretrained_model, self).__init__()


        # Select an optimizer for your network

        self.optimizer = tf.keras.optimizers.Adam()
        self.model_name = model_name
        
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
            
        self.head = [
              # Flatten(),
              # Dense(512, activation='relu'),
              # Dense(128, activation='relu'),
              # Dense(15, activation='softmax')
            tf.keras.layers.Average()
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


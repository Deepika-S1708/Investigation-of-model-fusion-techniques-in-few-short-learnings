import io
import os
import re
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import hyperparameters as hp

class CustomModelSaver(tf.keras.callbacks.Callback):
    """ Custom Keras callback for saving weights of networks. """

    def __init__(self, checkpoint_dir, task, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.task = task
        self.max_num_weights = max_num_weights
        self.max_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        """ At epoch end, weights are saved to checkpoint directory. """

        min_acc_file, max_acc_file, max_acc, num_weights = \
            self.scan_weight_files()

        cur_acc = logs["val_sparse_categorical_accuracy"]

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > self.max_acc:
            save_name = "{}_{}.weights.h5".format(
                epoch, cur_acc)

            
            save_location = self.checkpoint_dir + os.sep + "model." + save_name
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                   "maximum TEST accuracy.\nSaving checkpoint at {location}")
                   .format(epoch + 1, cur_acc, location = save_location))
            # Only save weights of classification head of VGGModel
            self.model.head.save_weights(save_location)

            # For fusion III
                
            # save_location = self.checkpoint_dir + os.sep + "model_head1." + save_name
            # self.model.head1.save_weights(save_location)
            
            # save_location = self.checkpoint_dir + os.sep + "model_head2." + save_name
            # self.model.head2.save_weights(save_location)
            
            # save_location = self.checkpoint_dir + os.sep + "model_head3." + save_name
            # self.model.head3.save_weights(save_location)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if self.max_num_weights > 0 and \
                    num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir + os.sep + min_acc_file)

                # For fusion III
                # os.remove(self.checkpoint_dir + os.sep + min_acc_file.replace('model.', 'model_head1.'))
                # os.remove(self.checkpoint_dir + os.sep + min_acc_file.replace('model.', 'model_head2.'))
                # os.remove(self.checkpoint_dir + os.sep + min_acc_file.replace('model.', 'model_head3.'))
                
            self.max_acc = cur_acc
        else:
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) DID NOT EXCEED "
                   "previous maximum TEST accuracy.\nNo checkpoint was "
                   "saved").format(epoch + 1, cur_acc))


    def scan_weight_files(self):
        """ Scans checkpoint directory to find current minimum and maximum
        accuracy weights files as well as the number of weights. """

        min_acc = float('inf')
        max_acc = 0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        files = os.listdir(self.checkpoint_dir)

        for weight_file in files:
            if weight_file.endswith(".h5"):
                num_weights += 1
                file_acc = float(re.findall(
                    r"[+-]?\d+\.\d+", weight_file.split("acc")[-1])[0])
                if file_acc > max_acc:
                    max_acc = file_acc
                    max_acc_file = weight_file
                if file_acc < min_acc:
                    min_acc = file_acc
                    min_acc_file = weight_file

        return min_acc_file, max_acc_file, max_acc, num_weights

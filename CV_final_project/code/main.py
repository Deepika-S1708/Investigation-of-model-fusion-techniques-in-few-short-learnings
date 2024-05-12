
import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import Pretrained_model
from preprocess import Datasets
from skimage.transform import resize
from ModelSaver import \
        CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.config.run_functions_eagerly(False)

def parse_args():
    """ Performing command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--model_name',
        required=True,
        choices=['resnet', 'densenet', 'efficientnet', 'inception', 'mobilenet', 'vgg', 'nasnet', 'ensemble'],
        help='''model name.''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default='test/Bedroom/image_0003.jpg',
        help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


def LIME_explainer(model, model_name, path, preprocess_fn, timestamp):
    """
    This function takes in a trained model and a path to an image and outputs 4
    visual explanations using the LIME model
    """

    save_directory = "lime_explainer_images" + os.sep + model_name + os.sep + timestamp + os.sep + path.split('/')[-2]
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists("lime_explainer_images" + os.sep + model_name):
        os.mkdir("lime_explainer_images" + os.sep + model_name)
    if not os.path.exists("lime_explainer_images" + os.sep + model_name + os.sep + timestamp):
        os.mkdir("lime_explainer_images" + os.sep + model_name + os.sep + timestamp)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        nonlocal image_index

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)

        image_save_path = save_directory + os.sep + str(image_index) + '_' + path.split('/')[-1]
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        image_index += 1

    # Read the image and preprocess it as before
    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = preprocess_fn(image)
    

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)


    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")

    image_save_path = save_directory + os.sep + str(image_index) + '_' + path.split('/')[-1]
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.show()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        CustomModelSaver(checkpoint_path, ARGS.task, 1)
    ]

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        init_epoch = int(ARGS.load_checkpoint.split('/')[7].split('_')[0][6:]) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    
    # set relative to the directory of main.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)


    # Run script from location of main.py
    os.chdir(sys.path[0])
    
    print(ARGS.data)

    datasets = Datasets(ARGS.data, ARGS.task)
    
        
    model = Pretrained_model(ARGS.model_name)
    checkpoint_path = "checkpoints/"+"{}_model".format(ARGS.model_name) + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "{}_model".format(ARGS.model_name) + os.sep + timestamp + os.sep
    
    # Print summaries of the model
    model(tf.keras.Input(shape=(224, 224, 3)))
    

    model.summary()


    # Load weights for fusion methods
    # model.head1.load_weights('checkpoints/vgg_model/050624-173657/model.27_0.7584589719772339.weights.h5')
    # model.head2.load_weights('checkpoints/resnet_model/050224-134004/model.29_0.7708542943000793.weights.h5')
    # model.head3.load_weights('checkpoints/efficientnet_model/050224-135859/model.49_0.7628140449523926.weights.h5')

    
    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
        

    if ARGS.evaluate:
        model.head.load_weights(ARGS.load_checkpoint)
        

        # For fusion methods
        # model.head1.load_weights(ARGS.load_checkpoint.replace('model.', 'model_head1.'))
        # model.head2.load_weights(ARGS.load_checkpoint.replace('model.', 'model_head2.'))
        # model.head3.load_weights(ARGS.load_checkpoint.replace('model.', 'model_head3.'))
            
            
        test(model, datasets.test_data)


        path = ARGS.lime_image
        LIME_explainer(model, ARGS.model_name, path, datasets.preprocess_fn, timestamp)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()

# Investigation-of-model-fusion-techniques-in-few-short-learnings


## Description
In this extensive research endeavor, our foremost objective is to meticulously examine the transferability of knowledge within deep neural networks. We aim to investigate the performance of various CNN pre-trained models under different data availability scenarios, with few-shot learning setups. We introduce various model fusion strategies to investigate the performance of different CNN models in a few short learning and very limited data for training. Our focus lies in determining the extent to which a model's learned representations in one domain can be effectively leveraged in a few short learning techniques. By systematically varying the amount of data provided during fine-tuning, we seek to uncover the threshold at which pre-trained models begin to exhibit competitive performance on target datasets distinct from their source domain. Through meticulous experimentation, we endeavor to elucidate the intricate interplay between a few short learning constraint data , model architecture, and transfer learning efficacy. This multifaceted approach will not only contribute to our understanding of deep neural network generalization but also provide valuable insights for optimizing model adaptation strategies across diverse domains.

## Project Structure
The project consists of the following files:

1. **main.py**: This file serves as the entry point to your project. It contain code for orchestrating the execution of the preprocessing, model training, and evaluation processes. 
2. **model.py**: This file contains the definition of machine learning model and fusion model architectures. It include the architecture of the model, such as layers, activation functions, and optimization techniques.
3. **preprocess.py**: This file contains code for preprocessing the dataset. It include functions for cleaning, transforming, and preparing the data for training and testing the model.
4. **ModelSaver.py**: This file contains code for saving trained models. It provide a function as callback function and could be used during model fitting.
5. **hyperparameters.py**: This file contains the hyperparameters used for our project, including image size, learning rate, epochs, etc.

## Usage

For base model fine-tune, simply changing model name for different model architectures.
```python
!python main.py --model_name 'vgg'
```

For ensemble model, there are some configuration in main.py and model.py need to be set for specific fusion strategy.
```python
!python main.py --model_name 'ensemble'
```


# %% [markdown]
# # Weather Prediction using Neural Network
# 
# Personal project as part of my portfolio.

# %% [markdown]
# # Table of Contents
# 
# 1. [Introduction](#introduction)
# 2. [Packages](#packages)
# 3. [Load the data](#load-data)
# 4. [Preprocess the data](#preprocess-data)
# 5. [Build the model](#model)
# 5. [Pipeline](#pipeline)
# 
# 

# %% [markdown]
# 

# %% [markdown]
# # Introduction <a class="anchor" id="introduction"></a>

# %% [markdown]
# # 1 - Packages <a class="anchor" id="packages"></a>

import glob
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
import pandas as pd
import pandas as pd
import seaborn as sns
# %%
import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras.layers as tfl
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# %% [markdown]
# ## 2 - Load the data <a class="anchor" id="load-data"></a>

# %% [markdown]
# ### Steps:
# - Make a list of all the images in the directory (path images)
# - Split path to get the name of the folder (label) and store it in a list (labels)
# - With labels and images, create a dataframe (data)

# %%
def load_img_data_to_df(path, img_extension='.jpg'):
    """
    Loads images from a directory and returns a dataframe with the image path and the label.
    """
    img_paths = (glob.glob(path + '/**/*'))
    labels = [x.split(os.sep)[-2] for x in img_paths]
    file_path = pd.Series(img_paths)
    labels = pd.Series(labels)
    df = pd.DataFrame({'File_Path': file_path, 'Labels': labels})
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df


# %%
path = 'dataset/'
data = load_img_data_to_df(path)
data.head()


# %% [markdown]
# # 2 - EDA - Exploratory Data Analysis <a class="anchor" id="eda"></a>

# %% [markdown]
# ### Steps:
# 
# - Check the dataframe (data)
# - Check the distribution of the data
# - Plot some images from the data

# %%
def plot_image_and_labels(data: pd.DataFrame, n_rows: int = 4, n_cols: int = 4, figsize: tuple = (10, 10)):
    """
    Show images and labels in a grid of n_rows x n_cols. Dataframe must have the columns 'File_Path' and 'Labels'.
    Args:
        data: Dataframe with the columns 'File_Path' and 'Labels'
        n_rows (opt): Number of rows in the grid
        n_cols (opt): Number of columns in the grid
        figsize (opt): Figure size
    """
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, ax in enumerate(ax.flat):
        img = plt.imread(data['File_Path'][i])
        ax.imshow(img)
        ax.set(title=data['Labels'][i])
        ax.axis('off')


# %%
def plot_label_distribution(data: pd.DataFrame):
    """
    View the distribution of labels in the dataframe.
    """
    sns.set_theme(style="darkgrid")
    sns.countplot(x='Labels', data=data, alpha=0.8, order=data['Labels'].value_counts().index)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=50)
    plt.show()


# %%
plot_image_and_labels(data)

# %%
plot_label_distribution(data)


# %% [markdown]
# The data is imbalanced, so it would be great to use some methods to balance the data. Not in the scope of this project.

# %% [markdown]
# # 3 - Preprocess Data <a class="anchor" id="preprocess-data"></a>
# 
# We use the ImageDataGenerator to generate batches of images and apply transformations to them before feeding them to the neural network. Image augmentation is a technique that is used to increase the number of training examples, as the dataset is too small to train the network on.

# %%
def image_generator_from_df(data, preprocessing_function=None, batch_size=32, image_size=(224, 224), seed=42):
    """
    Generates batches of tensor image data with real-time data augmentation from dataframe.
    Args:
        data: Dataframe with the columns 'File_Path' and 'Labels'
        preprocessing_function (opt): Function to preprocess the image data. Important to use the same preprocessing function if you want to use transfer learning.
        batch_size (opt): Batch size. Default is 32.
        image_size (opt): Image size. Default is (224,224).
        seed (opt): Random seed. Default is 42. Needed to make sure that the same data is used for training and validation.
    """
    train_datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocessing_function)
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=data,
        x_col='File_Path',
        y_col='Labels',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training',
        seed=seed,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=20,
    )
    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=data,
        x_col='File_Path',
        y_col='Labels',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation',
        seed=seed,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=20,
    )
    return train_gen, valid_gen


# %% [markdown]
# # 4 - Build the Model <a class="anchor" id="model"></a>
# 
# As we are implementing a CNN for image recognition and the training dataset is small, it could be useful to retrain a model from the Keras library as a starting point. The model will already be trained on the low-level features of the images, so we can replace last layers to classify on our custom labels.

# %%
from tensorflow.keras.applications import EfficientNetB7, InceptionV3, MobileNetV2, ResNet50, VGG19, Xception
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception


# %%
def get_pretrained_model(model, input_shape, n_classes, dropout_rate: int | None = None, preprocessing_function=None):
    """
    Adapts and add last layers to a pretrained image classification model. 
    Args:
        model: Pretrained model to use from get_base_model()
        input_shape: Input shape of the model.
        dropout_rate (opt): Dropout rate. If None, no dropout is used.
        n_classes: Number of classes for output layer
    """
    inputs = tf.keras.Input(shape=input_shape)
    # Apply preprocessing function to input
    if preprocessing_function is not None:
        inputs = preprocessing_function(inputs)
    x = model(inputs, training=False)
    # Use global average pooling to get the final feature vector
    x = tfl.GlobalAveragePooling2D()(x)
    # Add a dense layer with relu activation
    x = tfl.Dense(64, activation='relu')(x)
    # If dropout rate is not None, add dropout layer
    if dropout_rate is not None:
        x = tfl.Dropout(dropout_rate)(x)
    # Add a fully connected layer with n_classes neurons
    outputs = Dense(n_classes, activation='softmax')(x)
    # Create model
    model = Model(inputs, outputs)
    # Compile model
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


# %%
def get_base_model(model_name, input_shape):
    """
    Returns model based on the model_name and preprocessing function.
    Args:
        model_name: Name of the model.
        input_shape: Shape of the input.
    """
    if model_name == 'EfficientNetB7':
        model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocessing_function = preprocess_input_efficientnet
    elif model_name == 'InceptionV3':
        model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocessing_function = preprocess_input_inception
    elif model_name == 'MobileNetV2':
        model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocessing_function = preprocess_input_mobilenet
    elif model_name == 'ResNet50':
        model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocessing_function = preprocess_input_resnet
    elif model_name == 'VGG19':
        model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocessing_function = preprocess_input_vgg
    elif model_name == 'Xception':
        model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocessing_function = preprocess_input_xception
    else:
        raise ValueError('Model name not recognized.')
    model.trainable = False
    return model, preprocessing_function


# %%
def plot_history(history):
    """
    Plots the history of the training and validation accuracy and loss.
    Args:
        history: History object returned by model.fit()
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# %% [markdown]
# # 5 - Pipeline <a class="anchor" id="pipeline"></a>

# %%
# 0. Configuration variables
from tabnanny import verbose

path = 'dataset/'
BATCH_SIZE = 32
IMAGE_SIZE = (120, 120)
EPOCHS = 10
models = ['EfficientNetB7', 'MobileNetV2', 'ResNet50', 'VGG19', 'Xception']
# 1. Load data
data = load_img_data_to_df(path)
# 2. Split data into train and test and apply preprocessing
train, test = image_generator_from_df(data, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

# %%
# ----------------------------------------------------------------------------------------------------------------------
# 3. Train and evaluate models
models_dict = {}
for model_name in models:
    model_path = 'models/' + model_name + '.h5'
    # 3.1. Get base model
    model, pre_function = get_base_model(model_name, input_shape=IMAGE_SIZE + (3,))
    # 3.2. Adapt pretrained model
    model = get_pretrained_model(model, preprocessing_function=pre_function, input_shape=IMAGE_SIZE + (3,),
                                 n_classes=len(data['Labels'].unique()), dropout_rate=0.2)
    # 3.3. Train model
    my_callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')]
    history = model.fit(train, epochs=EPOCHS, validation_data=test, callbacks=my_callbacks)
    # 3.4. Evaluate model
    results = model.evaluate(test)
    # 3.4. Save model
    model.save(model_path)
    # 3.5 Store model and history in a dictionary
    models_dict[model_name] = {'model': model, 'history': history, 'results': results}

# %% [markdown]
# # Reports <a class="anchor" id="reports"></a>

# %%
import pickle


def save_results(models_dict):
    history_dict = {}
    for model_name in models:
        history_dict[model_name] = models_dict[model_name]['history']
        print(model_name + ': ' + str(models_dict[model_name]['results']))
    with open('models/modelsHistoryDict', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)


# %%
try:
    best_model = max(models_dict, key=lambda k: models_dict[k]['results'][1])
except:
    best_model = tf.keras.models.load_model('models/ResNet50.h5')
print('Best model:', best_model)
print('Best model accuracy:', models_dict[best_model]['results'][1])
plot_history(models_dict[best_model]['history'])

# %%
results_df = pd.DataFrame.from_dict(models_dict, orient='index')
results_df['Accuracy'] = results_df['results'].apply(lambda x: x[1])
results_df['Loss'] = results_df['results'].apply(lambda x: x[0])
results_df = results_df.drop('results', axis=1)
results_df = results_df.sort_values(by='Accuracy', ascending=False)

# Set figure size
plt.figure(figsize=(12, 8))
sns.barplot(y='Accuracy', x=results_df.index, data=results_df, palette='husl', orient='v')


# %%
def plot_classification_report(model, test_data):
    """
    Plots classification report from sklearn and a few predictions from test data
    """
    pred = model.predict(test_data)
    pred = np.argmax(pred, axis=1)
    # Get test data labels
    labels = test_data.class_indices
    labels = {v: k for k, v in labels.items()}
    pred = [labels[k] for k in pred]
    test_labels = [labels[k] for k in test_data.labels]
    clr = classification_report(test_labels, pred)
    print(clr)
    cf = confusion_matrix(test_labels, pred)
    # Confusion matrix
    ax = plt.subplot()
    sns.heatmap(cf, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels.values(), rotation=50)
    ax.yaxis.set_ticklabels(labels.values(), rotation=0)
    # Display x pictures with their predictions
    fig, ax = plt.subplots(4, 4, figsize=(14, 12))
    for i, ax in enumerate(ax.flat):
        ax.imshow(test_data[0][0][i].astype(np.uint8))
        ax.set_title(f'True label: {test_labels[i]}\nPredicted label: {pred[i]}')
        ax.axis('off')


plot_classification_report(models_dict[best_model]['model'], test)

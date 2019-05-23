import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

print("Tensorflow version is:")
print(tf.__version__)

# config settings

# The directories where the images are stored
data_dir = "../../data/kandinsky/MoreRedThanBlueSparse/42/"
train_dir = os.path.join(data_dir, "train/")
test_dir = os.path.join(data_dir, "test/")

# training settings
epochs = 20
steps_per_epoch = 100
batch_size = 20

# settings for plot naming
dataset_name = "MRtB, train/test overlap"


def plotFileName():
    return str(int(generator_train.n / 1000)) + "k_" + str(epochs) + "ep_block4_5"


# Helper-function for joining a directory and list of filenames
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


# Helper-function for plotting images
#
# Function used to plot at most 9 images in a 3x3 grid, and writing the true and predicted classes below each image.

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.savefig("../../findings/plots/wrongPredictions/" + plotFileName() + "_falsepred.png")
    plt.show()

# Helper-function for printing confusion matrix
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    print("Confusion matrix:")

    # Print the confusion matrix as text.
    print(cm)

    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))


# Helper-function for plotting example errors
#
# Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_test)

    # Get the file-paths for images that were incorrectly classified.
    image_paths = np.array(image_paths_test)[incorrect]

    # Load the first 9 images.
    images = load_images(image_paths=image_paths[0:9])

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the 9 images we have loaded and their corresponding classes.
    # We have only loaded 9 images so there is no need to slice those again.
    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# Function for calculating the predicted classes of the entire test-set and calling the above function to plot a few examples of mis-classified images.
def example_errors():
    # The Keras data-generator for the test-set must be reset
    # before processing. This is because the generator will loop
    # infinitely and keep an internal index into the dataset.
    # So it might start in the middle of the test-set if we do
    # not reset it first. This makes it impossible to match the
    # predicted classes with the input images.
    # If we reset the generator, then it always starts at the
    # beginning so we know exactly which input-images were used.
    generator_test.reset()

    # Predict the classes for all images in the test-set.
    y_pred = new_model.predict_generator(generator_test,
                                         steps=steps_test)

    # Convert the predicted classes from arrays to integers.
    cls_pred = np.argmax(y_pred, axis=1)

    # Plot examples of mis-classified images.
    plot_example_errors(cls_pred)

    # Print the confusion matrix.
    print_confusion_matrix(cls_pred)


# Helper-function for loading images
#
# The data-set is not loaded into memory, instead it has a list of the files for the images in the training-set and another list of the files for the images in the test-set.
# This helper-function loads some image-files.

def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)


# Helper-function for plotting training history
#
# This plots the classification accuracy and loss-values recorded during training with the Keras API.

def plot_training_history(history, accuracy):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title(dataset_name + ", " + str(generator_train.n) + " train, " + str(generator_test.n) + " test, " + str(
        epochs) + "epochs, accuracy: " + str(accuracy))
    plt.legend()

    # Ensure the plot shows correctly.
    plt.savefig("../../findings/plots/" + plotFileName() + "_history.png")
    plt.show()


# Pre-Trained Model: VGG16
#
# The following creates an instance of the pre-trained VGG16 model using the Keras API. This automatically downloads the
# required files if you don't have them already. Note how simple this is in Keras compared to Tutorial #08.
#
# The VGG16 model contains a convolutional part and a fully-connected (or dense) part which is used for classification.
# If include_top=True then the whole VGG16 model is downloaded which is about 528 MB. If include_top=False then only the
# convolutional part of the VGG16 model is downloaded which is just 57 MB.
#
# We will try and use the pre-trained model for predicting the class of some images in our new dataset, so we have to
# download the full model, but if you have a slow internet connection, then you can modify the code below to use the
# smaller pre-trained model without the classification layers.

model = VGG16(include_top=True, weights=None)

# Input Pipeline
#
# The Keras API has its own way of creating the input pipeline for training a model using files.
#
# First we need to know the shape of the tensors expected as input by the pre-trained VGG16 model.
# In this case it is images of shape 224 x 224 x 3.
input_shape = model.layers[0].output_shape[1:3]
print(input_shape)

# Keras uses a so-called data-generator for inputting data into the neural network, which will loop over the data for
# eternity.
#
# We have a small training-set so it helps to artificially inflate its size by making various transformations to the
# images. We use a built-in data-generator that can make these random transformations. This is also called an augmented
# dataset.

datagen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.9, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# We also need a data-generator for the test-set, but this should not do any transformations to the images because
# we want to know the exact classification accuracy on those specific images. So we just rescale the pixel-values so
# they are between 0.0 and 1.0 because this is expected by the VGG16 model.
datagen_test = ImageDataGenerator(rescale=1. / 255)

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=None)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

steps_test = generator_test.n / batch_size
print("steps test:" + str(steps_test))

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes

class_names = list(generator_train.class_indices.keys())
print("classes:")
print(class_names)

num_classes = generator_train.num_classes
print("number classes")
print(str(num_classes))

from sklearn.utils.class_weight import compute_class_weight

class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)




def predict(image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    plt.imshow(img_resized)
    plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)

    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred)[0]

    # Print the predictions.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))


transfer_layer = model.get_layer('block5_pool')
conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)

# Start a new Keras Sequential model.
new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add the final layer for the actual classification.
new_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-3)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']


def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


# enter fine-tuning: lets train not only the last layer
conv_model.trainable = True

# We want to train the last two convolutional layers whose names contain 'block5' or 'block4'.

for layer in conv_model.layers:
    # Boolean whether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)

    # Set the layer's bool.
    layer.trainable = 1

# check if correct layers are trainable

print_layer_trainable()

# low training rate for those layers
optimizer_fine = Adam(lr=1e-5)

# recompile model to apply 'trainable' changes
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

# retrain the model

history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

result = new_model.evaluate_generator(generator_test, steps=steps_test)
plot_training_history(history, "{0:.2%}".format(result[1]))
print("Test-set classification accuracy: {0:.2%}".format(result[1]))
example_errors()

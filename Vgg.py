import keras.applications.resnet
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Dropout, Activation
# from tf.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import glob, os, cv2, random, time
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50


def processing_data(data_path):
    """
    processing data
    :param data_path: training set path
    :return: train, val, test: processed training set, validation set, testing set
    """
    train_data = ImageDataGenerator(
        rescale=1. / 225,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        # Proportion of training data used as validation set
        validation_split=0.1
    )
    # generate validation data
    validation_data = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.1)
    train_generator = train_data.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='training',
        seed=0)
    validation_generator = validation_data.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        seed=0)

    test_data = ImageDataGenerator(
        rescale=1. / 255
    )

    test_generator = test_data.flow_from_directory(
        "./test",
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        seed=0,
    )
    return train_generator, validation_generator, test_generator


def model(train_generator, validation_generator, save_model_path):
    #resnet50_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(150, 150, 3))
    vgg16_model = VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(6, activation='softmax'))

    model = Sequential()
    model.add(vgg16_model)
    model.add(top_model)

    # complie model, using compile function: https://keras.io/models/model/#compile
    model.compile(
        # optimizer: Adam、sgd、rmsprop etc.
        optimizer=SGD(lr=1e-3, momentum=0.9),
        # loss function, using categorical_crossentropy

        loss='categorical_crossentropy',
        #
        metrics=['accuracy'])

    model.fit_generator(
        # generate instance
        generator=train_generator,
        # epochs:
        epochs=200,
        # The number of steps in an epoch, which should usually be equal to the number of samples in your dataset divided by the batch size
        steps_per_epoch=2048 // 16,
        # validation set
        validation_data=validation_generator,
        # On the validation set, the number of steps contained in an epoch should generally be equal to the number of samples in your dataset divided by the batch size.
        validation_steps=220 // 16,
    )
    model.save(save_model_path)

    return model


def evaluate_mode(test_generator, save_model_path):
    # load model
    model = load_model('results/myvgg16.h5')
    # get validation set loss and accuracy
    loss, accuracy = model.evaluate_generator(test_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))


def predict(img):
    """
    load model and use model to predict image
    steps:
        1.load model
        2.image process
        3.predict image label
    :param img: PIL.Image
    :return: string, image label,
            from 'cardboard','glass','metal','paper','plastic','trash' six classes
    """
    # turn image into numpy
    img = img.resize((150, 150))
    img = image.img_to_array(img)

    # load model,  model_path is relative path
    #  model_path = 'results/mymodel.h5'
    model_path = 'results/myvgg16.h5'
    '''
    try:

        model_path = os.path.realpath(__file__).replace('main.py', model_path)
    except NameError:
        model_path = './' + model_path
    '''
    # -------------------------- model predict code---------------------------
    # load model
    model = load_model(model_path)

    # expand_dims turn img.shape to (1, img.shape[0], img.shape[1], img.shape[2])
    x = np.expand_dims(img, axis=0)

    # model predict
    y = model.predict(x)

    # get labels
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

    # -------------------------------------------------------------------------
    predict = labels[np.argmax(y)]

    # return picture class
    return predict


def main():
    """

    the code includes data process, build model, train model, save model, evaluate model and model predict
    :return:
    """
    data_path = "./dataset-resized"
    save_model_path = 'results/myvgg16.h5'  # model path
    # get data
    train_generator, validation_generator, test_generator = processing_data(data_path)
    # build, train and save model
    model(train_generator, validation_generator, save_model_path)
    # elvaluate model
    evaluate_mode(test_generator, save_model_path)


if __name__ == '__main__':
    main()

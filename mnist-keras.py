# Adapted from:
# https://towardsdatascience.com/auto-keras-or-how-you-can-create-a-deep-learning-model-in-4-lines-of-code-b2ba448ccf5e

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def prepare_data():
    # Load MNIST and split to train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test

class DigitClassifier(object):
    def __init__(self, batch_size=128, num_classes=10, epochs=12, img_dim=(28, 28)):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.img_rows, self.img_cols = img_dim

    def reshape_inputs(self, x_train, x_test):
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)

            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)

        return x_train, x_test, input_shape

    def convert_to_matrices(self, y_train, y_test):
        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return y_train, y_test

    def create_model(self, input_shape):
        model = Sequential()

        # Conv > Conv > MaxPool > DO > Flat > Dense > DO > Dense
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

        return model

    def fit_and_score(self, input_shape, x_train, y_train, x_test, y_test):
        model = self.create_model(input_shape)
        model.fit(x_train, y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=1,
              validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)

        return score


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_data()

    dc = DigitClassifier()
    x_train, x_test, input_shape = dc.reshape_inputs(x_train, x_test)
    y_train, y_test = dc.convert_to_matrices(y_train, y_test)

    score = dc.fit_and_score(input_shape, x_train, y_train, x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


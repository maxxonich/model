#%%
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import EarlyStopping

model_name='digit_model.h5'


# load dataset, split to train and test
def load_dataset(count):
    if count==-1:
        # load dataset
        (trainX, trainY), (testX, testY) = mnist.load_data()
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        return trainX, trainY, testX, testY
    else:
        # load dataset
        (trainX, trainY), (testX, testY) = mnist.load_data()
        trainX=trainX[0:count,:,:]
        trainY=trainY[0:count]
        testX=testX[0:1000,:,:]
        testY=testY[0:1000]
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        return trainX, trainY, testX, testY



# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# define cnn model
def define_model():
    # model 1
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(10, activation='softmax'))
    # # compile model
    # opt = SGD(learning_rate=0.01, momentum=0.9)
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # return model


    # model 2
    model = Sequential()
    # model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer=RandomNormal(mean=0., stddev=1.), input_shape=(28, 28, 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer="he_uniform"))
    # model.add(Conv2D(64, (3, 3), activation='softmax', kernel_initializer=RandomNormal(mean=0., stddev=1.)))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # model.add(Dense(750, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(784, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(588, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(196, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = Adam(learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam')
    # opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# run the test harness for evaluating a model
def run_train(n):
    # load dataset
    trainX, trainY, testX, testY = load_dataset(n)
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # fit model
    # model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=1)

    model.fit(trainX, trainY, epochs=5, batch_size=32, verbose=1,callbacks=
    [EarlyStopping(monitor='loss', patience=3,verbose=1,restore_best_weights=True)]
              )
    # save model
    model.save(model_name)

def run_test(n):
    trainX, trainY, testX, testY = load_dataset(n)
    model= load_model(model_name)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('accuracy= %.3f' % (acc * 100.0))

#%%
# entry point
run_train(-1)
#%%
run_test(-1)
#%%
from scipy.ndimage import center_of_mass
import math
import cv2
import numpy as np

def getBestShift(img):
    cy, cx = center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


from PIL import Image as im

def rec_digit(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # im.fromarray(img).show()

    gray= cv2.Canny(image=img, threshold1=100, threshold2=200)  # Canny Edge Detection
    # gray = 255 - img
    # im.fromarray(gray).show()

    # im.fromarray(gray).show()
    #apply thresholding
    # (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # delete zero rows and cols
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)
    rows, cols = gray.shape

    # im.fromarray(gray).show()

    # change size to put into the box 20x20
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows),interpolation=cv2.INTER_AREA)
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows),interpolation=cv2.INTER_AREA)

    # im.fromarray(gray).show()

    # change size to 28x28
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
    # im.fromarray(gray).show()

    # moving mass center
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted
    # im.fromarray(gray).show()

    cv2.imwrite('gray' + img_path, gray)
    img = gray / 255.0
    img = np.array(img).reshape(-1, 28, 28, 1)

    model= load_model(model_name)
    out = str(np.argmax(model.predict(img)))
    return out

#%%
for i in range(10):
    filename="validation/"+str(i)+"_1png.png"
    print("File:",filename,"\nResult: ",rec_digit(filename))


# %%
def different_shapes_test():
    for i in [2000,3000,4000,6000,8000]:
        print("Size of dataset= ",i)
        run_train(i)
        run_test(i)
different_shapes_test()
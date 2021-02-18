import pandas as pd
import numpy as np
import keras
from sklearn import model_selection
from keras.models import Sequential

df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X_test= df_test.values.reshape(-1, 28, 28, 1)
X_test = X_test.astype(np.float32) / 255

X_test.shape

n_classes = 10

Y_train= df_train['label']
X_train= df_train.drop(columns='label')

Y_train = keras.utils.to_categorical(Y_train, n_classes)

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_train = X_train.astype(np.float32) / 255

x_train,x_test,y_train,y_test = model_selection.train_test_split(X_train,Y_train,test_size =0.2)

from keras import backend as K


def hard_swish(x):
    return x * (K.relu(x + 3., max_value = 6.) / 6.)


def swish(x):
    return x * K.sigmoid(x)
    
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers import Activation, Conv2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(10, (3,3), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(10, (3,3), padding='same',activation = hard_swish))
model.add(MaxPool2D())
model.add(Conv2D(10, (3,3), padding='same',activation = hard_swish))
model.add(Conv2D(10, (3,3), padding='same',activation = hard_swish))
model.add(Conv2D(10, (3,3), padding='same',activation = hard_swish))
model.add(MaxPool2D())
model.add(Conv2D(10, (3,3), padding='same',activation = hard_swish))
model.add(Conv2D(10, (3,3), padding='same',activation = hard_swish))
model.add(Conv2D(10, (3,3), padding='same',activation = hard_swish))
model.add(MaxPool2D())
model.add(Conv2D(10, (5,5), padding='same',activation = hard_swish))
model.add(Conv2D(10, (5,5), padding='same',activation = hard_swish))
model.add(Conv2D(10, (5,5), padding='same',activation = hard_swish))
model.add(MaxPool2D())
model.add(Conv2D(10, (5,5), padding='same',activation = hard_swish))
model.add(Conv2D(10, (5,5), padding='same',activation = hard_swish))
model.add(Conv2D(10, (5,5), padding='same',activation = hard_swish))
model.add(GlobalAveragePooling2D())

model.add(Dense(n_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128,
          epochs=80,
          verbose=1)
model.evaluate(x_test, y_test)

test_y_pred = model.predict(X_test)

nums_list = list()
for i in test_y_pred:
    nums_list.append(np.argmax(i))
    
import csv
with open('submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')
    writer.writerow(['ImageId', 'Label'])
    for i in range(len(nums_list)):
        writer.writerow([i+1,nums_list[i]])

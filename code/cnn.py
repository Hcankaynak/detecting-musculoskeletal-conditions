#from google.colab import drive
#drive.mount('/content/drive')

#!unzip "drive/My Drive/CNN/train_wrist.zip"

#!unzip "drive/My Drive/CNN/test_wrist.zip"

#importing necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
#from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import math

#creation of the training set and the test set 
train_datagen = ImageDataGenerator( rotation_range=30, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
                                    rescale=1./255,
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip=True,
                                    fill_mode = 'nearest',  
                                    validation_split = 0.068)

training_set = train_datagen.flow_from_directory('train_set_wrist/',
                                                 target_size = (224, 224),
                                                 batch_size = 8,
                                                 class_mode = 'categorical', subset="training")

validation_set = train_datagen.flow_from_directory('train_set_wrist/',
                                                 target_size = (224, 224),
                                                 batch_size = 8,
                                                 class_mode = 'categorical', subset="validation")

test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('test_set_wrist/',
                                            target_size = (224, 224),
                                            batch_size = 1,
                                            class_mode = 'categorical',
                                            shuffle = False)

#class_weights = class_weight.compute_class_weight('balanced',
                                                  #np.unique(training_set.classes), 
                                                  #training_set.classes)

#CNN Model implementation
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
#classifier.add(Dense(units = 64, activation = 'relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 2, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
check_point = ModelCheckpoint("drive/My Drive/CNN/cnn_utk_hand.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callback_list = [check_point]

classifier.summary()

history = classifier.fit_generator(training_set,
                         steps_per_epoch = len(training_set),
                         epochs = 100,
                         validation_data = validation_set,
                         validation_steps = len(validation_set), 
                         callbacks = callback_list)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#prediction part
Y_pred = classifier.predict_generator(test_set, steps = len(test_set))
y_pred = np.argmax(Y_pred, axis=1)

print('Classification Report')
target_names = ['negative', 'positive']
print(classification_report(test_set.classes, y_pred, target_names=target_names))
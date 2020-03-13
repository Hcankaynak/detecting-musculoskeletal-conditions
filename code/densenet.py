#from google.colab import drive
#drive.mount('/content/drive')

#!unzip "drive/My Drive/CNN/train_wrist.zip"

#!unzip "drive/My Drive/CNN/test_wrist.zip"

#importing necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet169
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
from keras.layers import MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, ReduceLROnPlateau
#from sklearn.utils import class_weight 
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.applications.densenet import DenseNet169
from keras.optimizers import RMSprop, Adam, SGD

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

#DenseNet model implementation
base_model = DenseNet169(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet', pooling = 'avg' )
x = base_model.output
predictions = Dense(2, activation='softmax')(x)
dense_model = Model(inputs=base_model.input, outputs=predictions)
dense_model.summary()

opt = Adam(lr = 0.0001)
dense_model.compile(optimizer = opt , loss= "binary_crossentropy", metrics = ['accuracy'])

check_p = ModelCheckpoint("drive/My Drive/CNN/dense_utku_hand_320.h5", monitor='val_loss', verbose=1, save_best_only= True, save_weights_only=False, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0, verbose = 1, cooldown=1)

callback_list = [check_p, reduce_lr]

history_dense = dense_model.fit_generator(training_set,
                         steps_per_epoch = len(training_set),
                         epochs = 25,
                         validation_data = validation_set,
                         validation_steps = len(validation_set), 
                         callbacks = callback_list,
                         class_weight = class_weights)

# Plot training & validation accuracy values
plt.plot(history_dense.history['acc'])
plt.plot(history_dense.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_dense.history['loss'])
plt.plot(history_dense.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#model = load_model("drive/My Drive/CNN/model_wrist_1.h5")
Y_pred = model.predict_generator(test_set, steps = len(test_set))
y_pred = np.argmax(Y_pred, axis=1)
print('Classification Report')
target_names = ['negative', 'positive']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

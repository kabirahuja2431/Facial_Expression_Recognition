
# coding: utf-8

# In[7]:

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16 
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization
import numpy as np


# In[48]:

img_width , img_height = 48,48
train_dir = 'data/train/'
test_dir = 'data/test/'
nb_train = 172254
nb_test = 43068
epochs = 10
batch_size = 32
input_shape = (img_width,img_height,3)
top_model_weights_path = 'bottleneck_fc_model.h5'

def vannila_model():
	model = Sequential()

	model.add(Conv2D(32,(3,3),input_shape = input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32,(3,3),input_shape = input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(BatchNormalization())
	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(BatchNormalization())
	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7))
	model.add(Activation('softmax'))

	model.compile(loss = 'categorical_crossentropy',
             	optimizer = 'rmsprop',
             	metrics = ['accuracy'])

	train_datagen = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

	test_datagen = ImageDataGenerator(rescale = 1./255)

	train_generator = train_datagen.flow_from_directory(
                	train_dir,
                	target_size = (img_width,img_height),
                	batch_size = batch_size,
                	class_mode='categorical')

	test_generator = test_datagen.flow_from_directory(
    					test_dir,
    					target_size=(img_width, img_height),
    					batch_size=batch_size,
    					class_mode='categorical')
	model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test // batch_size)

	model.save_weights('vannila.h5')


vannila_model()
# # Transfer Learning

# In[33]:

def save_features():
	datagen = ImageDataGenerator(rescale = 1./255)
	model = VGG16(weights = 'imagenet',include_top = False)
	generator = datagen.flow_from_directory(
				train_dir,
				target_size = (img_width,img_height),
				batch_size = batch_size,
				class_mode = 'categorical',
				shuffle = False)
	features_train = model.predict_generator(generator,nb_train//batch_size)
	np.save(open('features_train.npy','wb'),features_train)
	
	generator = datagen.flow_from_directory(
				test_dir,
				target_size = (img_width,img_height),
				batch_size = batch_size,
				class_mode = 'categorical',
				shuffle = False)
	features_test = model.predict_generator(generator,nb_test//batch_size)
	np.save(open('features_test.npy','wb'),features_test)


# In[34]:

#save_features()


# In[51]:

def transfer_learning():
	train_data = np.load(open('features_train.npy','rb'))
	train_labels = np.array([0]*1000 + [1]*101 + [2]*1022 + [3]*1776 + [4]*1168 + [5]*742 + [6]*1183)
	print(train_data.shape)
	print(train_labels.shape)
	test_data = np.load(open('features_test.npy','rb'))
	test_labels = np.array([0]*529 + [1]*63 + [2]*551 + [3]*989 + [4]*718 + [5]*458 + [6]*692)
	print(test_data.shape)
	print(test_labels.shape)
	model = Sequential()
	model.add(Flatten(input_shape = train_data.shape[1:]))
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7))
	model.add(Activation('softmax'))
	
	model.compile(loss = 'sparse_categorical_crossentropy',
				optimizer = 'rmsprop',
				metrics = ['accuracy'])
	
	model.fit(train_data,train_labels,
			 epochs = epochs,
			 batch_size = batch_size,
			 validation_data = (test_data,test_labels))
	
	model.save_weights(top_model_weights_path)


# In[ ]:

def fine_tune():
	model = VGG16(weights = 'imagenet',include_top = False,input_shape = (img_width,img_height,3))

	top_model = Sequential()
	print(model.output_shape[1:])
	top_model.add(Flatten(input_shape = model.output_shape[1:]))
	top_model.add(Dense(256))
	top_model.add(Activation('relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(7))
	top_model.add(Activation('softmax'))
	top_model.load_weights(top_model_weights_path)
	new_model = Sequential()
	for layer in model.layers:
		new_model.add(layer)
	new_model.add(top_model)
	for layer in new_model.layers[:25]:
		layer.trainable = False

	new_model.compile(loss = 'categorical_crossentropy',
				optimizer = optimizers.SGD(lr=1e-4,momentum = 0.9),
				metrics = ['accuracy'])

	train_gen = ImageDataGenerator(
				rescale = 1./255,
				shear_range = 0.2,
				horizontal_flip = True)

	test_gen = ImageDataGenerator(
				rescale = 1./255)

	train_generator = train_gen.flow_from_directory(
						train_dir,
						target_size = (img_height,img_width),
						batch_size = batch_size,
						class_mode = 'categorical')

	test_generator = test_gen.flow_from_directory(
						test_dir,
						target_size = (img_height,img_width),
						batch_size = batch_size,
						class_mode = 'categorical')

	new_model.fit_generator(
		train_generator,
		samples_per_epoch = nb_train,
		epochs = epochs,
		validation_data = test_generator,
		nb_val_samples = nb_test)

transfer_learning()
#fine_tune()
# In[ ]:




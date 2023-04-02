import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Dense,Dropout,Flatten

from tensorflow.keras.callbacks import EarlyStopping

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

brail_sample_org_data ="Dataset100/train"

brail_sample_org_data_list = sorted(os.listdir(brail_sample_org_data))
brail_number_of_classes = len(brail_sample_org_data)

resize_shape = (60,60)

def sort_directories(directory):
    new_dir_list = list()
    for file in os.listdir(directory):
        if file.split('.')[0].isdigit():
            new_dir_list.append(file)
    return sorted(new_dir_list, key=lambda f: int(f.split('.')[0]))

brail_sample_org_data_list = sort_directories(brail_sample_org_data)

brail_images = []
brail_class_names = []
brail_total_classes = []
for folder in brail_sample_org_data_list:  # root folder

    brail_data_sample_folder = os.listdir(str(brail_sample_org_data) + "/" + str(folder))

    for image in brail_data_sample_folder:  # samples inside the folder

        current_image = cv2.imread(brail_sample_org_data + "/" + str(folder) + "/" + str(image))  # image reading
        current_image = cv2.resize(current_image, resize_shape)  # image resizing
        brail_images.append(current_image)
        brail_class_names.append(folder)

    brail_total_classes.append(folder)

brail_images = np.array(brail_images)
brail_class_names = np.array(brail_class_names)
print("\n\nimages shape: " + str(brail_images.shape))
print("image classes shape: " + str(brail_class_names.shape))
print("No of image classes: " + str(len(brail_total_classes)))

#70:20:10
x_train,x_test,y_train,y_test = train_test_split(brail_images,brail_class_names,test_size=0.3)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=0.1)

print("x_train shape: "+str(x_train.shape))
print("y_train shape: "+str(y_train.shape))
print("x_validation shape: "+str(x_validation.shape))
print("y_validation shape: "+str(y_validation.shape))
print("x_test shape: "+str(x_test.shape))
print("y_test shape: "+str(y_test.shape))

def preprocess_image(img):
    if(len(img.shape)==3):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    normalized_img = img / 255
    return normalized_img

x_train = np.array(list(map(preprocess_image, x_train)))
x_test = np.array(list(map(preprocess_image, x_test)))
x_validation = np.array(list(map(preprocess_image, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

print("x_train shape: "+str(x_train.shape))
print("y_train shape: "+str(y_train.shape))
print("x_validation shape: "+str(x_validation.shape))
print("y_validation shape: "+str(y_validation.shape))
print("x_test shape: "+str(x_test.shape))
print("y_test shape: "+str(y_test.shape))

y_train = to_categorical(y_train, num_classes=brail_number_of_classes)
y_test = to_categorical(y_test, num_classes =brail_number_of_classes)
y_validation = to_categorical(y_validation, num_classes =brail_number_of_classes)

noOfFilters = 60
sizeOfFilter1 = (5, 5)
sizeOfFilter2 = (3, 3)
sizeOfPool = (2, 2)
noOfNodes = 500

model = Sequential()
model.add((Conv2D(noOfFilters, sizeOfFilter1,input_shape=(resize_shape[0], resize_shape[1], 1)))) # layer 01

model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))) # layer 02
model.add(MaxPooling2D(pool_size=sizeOfPool))

model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))) # layer 03

model.add(Flatten())
model.add(Dense(noOfNodes, activation='relu'))
model.add(Dense(brail_number_of_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

callbacks = [EarlyStopping(patience=1)]
history = model.fit(x_train,y_train,validation_data=(x_validation,y_validation), epochs=1000,callbacks=callbacks,verbose=0)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score=model.evaluate(x_test,y_test,verbose=0)
print('Test Accuracy = ',score[1])

# input
input_image = cv2.imread("TestingImages/2.jpg",cv2.IMREAD_COLOR)
resized_character = cv2.resize(input_image,resize_shape)
input_preprocessed_image = preprocess_image(resized_character)
reshaped_character = input_preprocessed_image.reshape(1,resize_shape[0],resize_shape[1],1)
prediction=model.predict(reshaped_character,verbose=0)
predicted_class=np.argmax(prediction,axis=1)

#matplotlib inline
plt.axis('off')
plt.imshow(input_image, cmap='gray')
plt.show()
print("predicted class: ",predicted_class)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools


import os
data_dir = (r'.\fire_dataset')
categories = ['non_fire_images', 'fire_images']
for i in categories:
    path = os.path.join(data_dir, i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))    


img_size = 128
image_array = cv2.resize(img_array, (img_size,img_size))



dno = cv2.imread('./fire_dataset/non_fire_images/non_fire.12.png')
dyes = cv2.imread('./fire_dataset/fire_images/fire.2.png')


train_data = []

for i in categories:
    train_path = os.path.join(data_dir,i)
    tag = categories.index(i)
    for img in os.listdir(train_path):
        try:
            image_arr = cv2.imread(os.path.join(train_path , img), cv2.IMREAD_GRAYSCALE)
            new_image_array = cv2.resize(image_arr, (img_size,img_size))
            train_data.append([new_image_array , tag])
        except Exception as e:
            pass


X = []
y = []
for i,j in train_data:
    X.append(i)
    y.append(j)
X = np.array(X).reshape(-1,img_size,img_size)
print(X.shape)
X = X/255.0  
X = X.reshape(-1,128,128,1)


from keras.utils.np_utils import to_categorical   

y_enc = to_categorical(y, num_classes = 4)




X_train , X_test, y_train, y_test = train_test_split(X , y_enc , test_size = 0.1, random_state = 42)
X_train , X_val, y_train, y_val = train_test_split(X_train , y_train , test_size = 0.1, random_state = 42)



from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#from keras.optimizers import RMSprop,Adam (if you use 2019 anaconda , please uncomment this line(remove "#") )
from tensorflow.keras.optimizers import RMSprop,Adam #(This line For 2020 anaconda. But If you use 2019 , comment this line with "#" )
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

model = Sequential()


model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (128,128,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(4, activation = "softmax"))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 5

es = EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience = 3
)

batch_size = 16
imggen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=0,
        zoom_range = 0,
        width_shift_range=0,  
        height_shift_range=0,  
        horizontal_flip=True,  
        vertical_flip=False)


from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("models/model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

imggen.fit(X_train)
history = model.fit_generator(imggen.flow(X_train,y_train,batch_size = batch_size),
                              epochs = epochs, validation_data = (X_val,y_val),
                              steps_per_epoch = X_train.shape[0] // batch_size,
                              callbacks=callbacks_list)



# serialize model structure to JSON
model_json = model.to_json()
with open(r".\models\firenet.data-00000-of-00001", "w") as json_file:
    json_file.write(model_json)

os.listdir(r"models")

import joblib
joblib.load("folder_name + extension")



import cv2
import os
import sys
import math
import datetime
import time

from firebase import firebase
import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression
fixefixed_interval = 3
firebase = firebase.FirebaseApplication('https://fire-detection-a75df-default-rtdb.firebaseio.com/', None)
count=1
def construct_firenet (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model


if __name__ == '__main__':


    # construct and display model

    model = construct_firenet (224, 224, training=False)
    print("Constructed FireNet ...")

    model.load(os.path.join("models", "firenet"),weights_only=True)
    print("Loaded CNN network weights ...")



    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - FireNet CNN";
    keepProcessing = True;


    if len(sys.argv) == 2:

        # load video file from first command line argument

        video = cv2.VideoCapture(sys.argv[1])
        print("Loaded video ...")

        # create window

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

        # get video properties

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps);

        while (keepProcessing):

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount();

            # get video frame from file, handle end of file

            ret, frame = video.read()
            if not ret:
                print("... end of video file reached");
                break;

            # re-size image to network input size and perform prediction

            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
            output = model.predict([small_frame])

            # label image based on prediction

            if round(output[0][0]) == 1:
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
                #print("Fire")
                if count==1:
                    datetime1=datetime.datetime.now()
                    date=datetime1.strftime("%x")
                    time=datetime1.strftime("%X")
                    day=datetime1.strftime("%A")
                    device = "42"
                    status="Fire"
                    data={"Device ID":device,"Status":status,"Date":date,"Time":time,"Day":day}
                    firebase.put('', 'FIRE DETECTION/Location 1', data)
                    #time.sleep(10)
                    status="clear"
                    data={"Device ID":device,"Status":status,"Date":date,"Time":time,"Day":day}
                    firebase.put('', 'FIRE DETECTION/Location 1', data)
                    count=0
                
            else:
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
                #print("Clear")
                if count==1:
                    
                    device = "42"
                    status="Clear"
                    data={"Device ID":device,"Status":status}
                    firebase.put('', 'FIRE DETECTION/Location 1', data)
                    count=0
                            
                
                
                

            # stop the timer and convert to ms. (to see how long processing and display takes)

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

            # image display and key handling

            cv2.imshow(windowName, frame);

            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            if (key == ord('x')):
                keepProcessing = False;
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    else:
        print("usage: python firenet.py videofile.ext");
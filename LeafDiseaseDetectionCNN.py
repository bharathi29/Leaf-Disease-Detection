#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
from keras_preprocessing.image import img_to_array, load_img
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm


# In[2]:


classnames = ['CherryHealthy','CherryPowderyMildew','PeachBacterialSpot','PeachHealthy','PepperBellBacterial','PepperBellHealthy','PotatoEarlyBlight','PotatoHealthy','PotatoLateBlight','TomatoHealthy','StrawberryHealthy','StrawberryLeafScorch','TomatoBacterialSpot','TomatoLateBlight','TomatoLeafMold','TomatoMosaicVirus','TomatoSeptoriaLeafSpot','TomatoSpidermitesTwospottedspidermite']
classnameslabel = {classname : i for i, classname in enumerate(classnames)}
nb_classes = len(classnames)
print(classnameslabel)
IMAGE_SIZE = (150,150)


# In[3]:


def load_data():
    DIRECTORY=r"D:\image\ar"
    CATEGORY=["Training","Testing"]
    
    output=[]
    
    for cat in CATEGORY:
        path=os.path.join(DIRECTORY, cat)
        images=[]
        labels=[]
        print("Loading {}".format(cat))
        for folder in os.listdir(path):
            label = classnameslabel[folder]
            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(os.path.join(path, folder), file)
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,IMAGE_SIZE)
                
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = "float32")
        labels = np.array(labels, dtype = "int32")

        output.append((images, labels))
        
    return output


# In[4]:


(trainim, trainlab), (testim, testlab) = load_data()


# In[5]:


trainim, trainlab = shuffle(trainim, trainlab, random_state=25)


# In[6]:


def display_examples(classnames, images, labels):
    figsize=(20,20)
    fig=plt.figure(figsize=figsize)
    fig.suptitle("Examples of some from the dataset: ", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        #image=cv2.resize(images[i],figsize)
        plt.imshow(images[i].astype(np.uint8))
        plt.xlabel(classnames[labels[i]])
    plt.show()
display_examples(classnames,trainim,trainlab)


# In[7]:


model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(19, activation=tf.nn.softmax)
])


# In[8]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[9]:


history = model.fit(trainim, trainlab, batch_size=128, epochs=2, validation_split = 0.2)


# In[10]:


def plot_accuracy_loss(history):
    
    fig=plt.figure(figsize=(10,5))
    
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--',label = "acc")
    plt.plot(history.history['val_accuracy'],'ro--',label = "val_loss")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--',label = "loss")
    plt.plot(history.history['val_loss'],'ro--',label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    
    plt.legend()
    plt.show()
    


# In[11]:


plot_accuracy_loss(history)


# In[12]:


test_loss=model.evaluate(testim,testlab)


# In[13]:


predictions = model.predict(testim)
pred_labels = np.argmax(predictions, axis = 1)
print(classification_report(testlab, pred_labels))


# In[14]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

model = VGG16(weights = 'imagenet',include_top = False)
model = Model(inputs = model.inputs, outputs = model.layers[-5].output)


# In[15]:


trainfeat = model.predict(trainim)
testfeat = model.predict(testim)


# In[16]:


from keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D, Flatten

model2= VGG16(weights = 'imagenet', include_top = False)

input_shape = model2.layers[-4].get_input_shape_at(0)
layer_input = Input(shape = (9, 9, 512))

x = layer_input
for layer in model2.layers[-4::2]:
    x = layer(x)
x = Conv2D(64, (3,3), activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Flatten()(x)
x = Dense(100, activation = 'relu')(x)
x = Dense(19, activation = 'softmax')(x)

new_model = Model(layer_input, x)


# In[17]:


new_model.compile(optimizer= "adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[18]:


history=new_model.fit(trainfeat, trainlab, batch_size=128, epochs=4, validation_split=0.2)


# In[19]:


plot_accuracy_loss(history)


# In[20]:


from sklearn.metrics import accuracy_score

predictions = new_model.predict(testfeat)
pred_labels = np.argmax(predictions, axis=1)


print("Accuracy : {}".format(accuracy_score(testlab, pred_labels)))


# In[21]:


print(classification_report(testlab, pred_labels))


# In[22]:


dictionary = dict(zip(list(classnameslabel.keys()), list(classnameslabel.values())))
print(dictionary)
allapairs=[]
allpairs=list(dictionary.items())
#print("Ref :",allpairs[0])


# In[23]:


def prediction(path):
    img = load_img(path, target_size= (150,150))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis= 0)
    pred+ = np.argmax(predictions)
    if(pred==0 or pred==3 or pred==5 or pred==7 or pred==9 or pred==10):
        print("Your leaf is healthy!!!")
    else:
        print("Your leaf is unhealthy:(")
    #print(f"The image belongs to {allpairs[pred]}")


# In[23]:


#DIR1=r"D:\tohe.jpg"
#path1=os.path.join(DIR1)

tf.keras.models.save_model(model, 'model.pbtxt')

converter = tf.lite.TFLiteConverter.from_keras_model(model = model)

model_tflite = converter.convert()

open("le.tflite","wb").write(model_tflite)


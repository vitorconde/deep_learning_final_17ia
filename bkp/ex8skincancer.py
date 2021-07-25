
!pip install tensorflow
!pip install opencv-contrib-python 
!pip3 install opencv-contrib-python 
!pip install keras

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout , Flatten, Activation
from keras.layers.convolutional import Conv2D , AveragePooling2D
from keras.layers.convolutional import MaxPooling2D 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import cv2
from keras.preprocessing.image import img_to_array , load_img
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.cm

df= pd.read_csv('C:/Users/vinic/Downloads/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv')

dfnlpy =  df.drop(['label'], axis=1).to_numpy()
dfnlpy= dfnlpy.reshape(len(df),28,28,3)
dfnlpy.shape

#lenn=len(df)
#for i in range(9):
#    plt.subplot(330+1+i)
#    rand= random.randint(1,lenn )
#    print( rand)
#    plt.imshow(dfnlpy[rand])
#plt.show()

def filterblackborders(object):
    ii =0
    for i  in object:        
        r=int(np.quantile(i[:,:,0],0.75))
        g=int(np.quantile(i[:,:,1],0.75))
        b=int(np.quantile(i[:,:,2],0.75))
        bright_channel =[r,g,b] # obtem o padrão claro de pixel na imagem
        i[0:28,22:28] =bright_channel
        i[22:28,0:28] =bright_channel
        i[0:28,0:6] =bright_channel
        i[0:6,0:28] =bright_channel
        ii=ii+1
        print(ii)

blackborders(dfnlpy)

df2 = pd.DataFrame(dfnlpy.reshape(10015,2352))
df2 =df2.join(df['label'])
dfnlpy = df2.to_numpy()
dfnlpy.shape
df2.shape


dfnlg= df2.groupby('label')
dftrain= pd.DataFrame()
dftest=pd.DataFrame()

for i,g in dfnlg:
    train,test= np.split(g,[int(len(g)*.8)])
    dftrain=dftrain.append(train, ignore_index=False);
    dftest=dftest.append(test, ignore_index=False);

    
dftrain.columns = df.columns
dftest.columns = df.columns


xtrain, xtest, ytrain, ytest = train_test_split( dfnlpy,df.label)
xtrain =  dftrain.drop(['label'], axis=1).to_numpy()
ytrain = dftrain['label'].to_numpy()
xtest  =  dftest.drop(['label'], axis=1).to_numpy()
ytest = dftest['label'].to_numpy()


img_rows, img_cols = 28,28
xtrain= xtrain.reshape(xtrain.shape[0], img_rows, img_cols,3)
xtest= xtest.reshape(xtest.shape[0],img_rows,img_cols,3)

xtrain=xtrain/255
xtest = xtest/255


ytrain = np_utils.to_categorical(ytrain)
ytest= np_utils.to_categorical(ytest)

numclasses = ytest.shape[1]
model_input_shape = (img_rows,img_cols,3)

model = Sequential()
model.add(Conv2D(16, (3, 3), padding ='same', activation = 'relu',input_shape = model_input_shape))
model.add(Conv2D(32, (3, 3), padding ='same', activation = 'relu',input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = None,padding = 'valid',data_format = None))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), padding ='same', activation = 'relu',input_shape = (32, 32, 3)))
model.add(Conv2D(64, (3, 3), padding ='same', activation = 'relu',input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = None,padding = 'valid',data_format = None))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(400))
model.add(Dense(120))
model.add(Dense(numclasses))
model.add(LeakyReLU(0.1))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

checkpointer = ModelCheckpoint(filepath='C:/Users/vinic/Downloads/skin-cancer/checkpoints/skincancer_val_acc.hdf5',  save_best_only=True, monitor='val_accuracy')

history= model.fit(xtrain, ytrain,  callbacks=[checkpointer], validation_data= (xtest,ytest),epochs=20,batch_size=10, shuffle=True)


plt.figure(1)
#plt.plot(history.history['categorical_accuracy'])
#plt.plot(history.history['val_categorical_accuracy'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','loss', 'validation', 'val_loss'], loc ='upper left')
plt.show()




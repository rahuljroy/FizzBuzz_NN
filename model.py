import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import plot_model

###############################################################################
#Generating the training data
#Training inputs
X_train = []
for i in range(101, 1001):
  res = []
  res  = [int(x) for x in list('{0:0b}'.format(i))]
  resnew = []
  length = len(res)
  if(length<10):
    for l in range(length+1,11):
      resnew.append(0)
  for i in res:
    resnew.append(i)
  X_train.append(resnew)

X_train = np.asarray(X_train)

#Training Outputs
y_train = []
for x in range(101, 1001):
  if int(x)%3 ==0 and int(x)%5 == 0:
    y_train.append(0)
  elif int(x)%3 ==0:
    y_train.append(1)
  elif int(x)%5 == 0:
    y_train.append(2)
  else:
    y_train.append(3)


###############################################################################
#Model
from keras.utils import to_categorical
y_train = to_categorical(y_train)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

classifier = Sequential()

classifier.add(Dense(1000, kernel_initializer='uniform', activation = 'relu', input_dim = 10))
classifier.add(Dropout(p=0.1))
#Adding the second hidden layer
classifier.add(Dense(1000, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(1000, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))
#Adding the output layer
classifier.add(Dense(4, kernel_initializer='uniform', activation = 'softmax'))

adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_split = 0.1)

#Plotting the statistics
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='lower right')
plt.show()

fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper right')
plt.show()
#Saving the model to disk
classifier.save("model.h5")
print("Model Saved")
###############################################################################

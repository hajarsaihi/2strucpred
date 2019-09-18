##### (1) Import Modules ###################################################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv1D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd

##### (2) Load Data ########################################################################################################
x_train = np.load(open('x_win17_seqpad_train_17k_3.npy', 'rb'), allow_pickle=True)
y_train = np.load(open('y_win17_seqpad_train_17k_3.npy', 'rb'), allow_pickle=True)
x_val = np.load(open('x_win17_seqpad_val.npy', 'rb'), allow_pickle=True)
y_val = np.load(open('y_win5_seqpad_val.npy', 'rb'), allow_pickle=True)

print ('x_train shape:', x_train.shape)
print ('y_train shape:', y_train.shape)
print ('x_val shape:', x_val.shape)
print ('y_val shape:', y_val.shape)

##### (3) Build Model ######################################################################################################
model = Sequential()
model.add(Conv1D(512, 3, input_shape=(15, 103), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(3, activation='softmax'))

model.summary()

opt = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.8)

##### (3a) Run Model #######################################################################################################
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, batch_size=100, validation_data=(x_val, y_val)) 
model.save('three_state_model.h5')

##### (4) Plot Graphs ######################################################################################################
history_dict = history.history

import matplotlib.pyplot as plt
plt.style.use('seaborn')

loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
epochs = range(1, len(acc) + 1)
acc = history_dict['acc']
val_acc = history_dict['val_acc']

fig, ax1 = plt.subplots(1)
ax1.plot(epochs, loss, 'b', label='Training loss', color='c')
ax1.plot(epochs, val_loss, 'b', label='Validation loss', color='blueviolet')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
fig.savefig('three_state_loss.jpeg') # save loss results as jpeg file

fig2, ax2 = plt.subplots(1)
ax2.plot(epochs, acc, 'b', label='Training acc', color='c')
ax2.plot(epochs, val_acc, 'b', label='Validation acc', color='blueviolet')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
fig2.savefig('three_state_acc.jpeg') # save accuracy results as jpeg file


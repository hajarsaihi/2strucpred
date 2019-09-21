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
x_train = np.load(open('/Users/0/Desktop/2strucpred/data/x_train.npy', 'rb'), allow_pickle=True) # this is for 17000 proteins
y_train = np.load(open('/Users/0/Desktop/2strucpred/data/y_train.npy', 'rb'), allow_pickle=True)
print ('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)

x_val = np.load(open('/Users/0/Desktop/2strucpred/data/x_val.npy', 'rb'), allow_pickle=True)
y_val = np.load(open('/Users/0/Desktop/2strucpred/data/y_val.npy', 'rb'), allow_pickle=True)
print ('x_val shape:', x_val.shape, 'y_val shape:', y_val.shape)

##### (3) Build Model ######################################################################################################
model = Sequential()
model.add(Conv1D(512, 3, input_shape=(15, 108), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(8, activation='softmax'))

model.summary()

opt = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.8)

##### (3a) Run Model #######################################################################################################
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=1, batch_size=100, validation_data=(x_val, y_val)) 
#model.save('eight_state_model.h5')

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
fig.savefig('eight_state_loss.jpeg') # save loss results as jpeg file

fig2, ax2 = plt.subplots(1)
ax2.plot(epochs, acc, 'b', label='Training acc', color='c')
ax2.plot(epochs, val_acc, 'b', label='Validation acc', color='blueviolet')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
fig2.savefig('eight_state_acc.jpeg') # save accuracy results as jpeg file

##### (4a) Conf Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

y_pred = model.predict([x_val])
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.75)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),size = 15, 
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('eight_confmat.jpeg')
    return ax


np.set_printoptions(precision=2)

plot_confusion_matrix(y_true, y_pred, classes=['B','E','G','H','I','S','T','Other'], normalize=True,
                      title='8-State Confusion Matrix on Test Data')
plt.show()


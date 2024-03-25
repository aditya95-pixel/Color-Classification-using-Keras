import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("final_data.csv")
dataset
# One-Hot-Encoding
dataset = pd.get_dummies(dataset, columns=['label'])
dataset.head()
train_dataset = dataset.sample(frac=0.8, random_state=8) #train_dataset = 80% of total dataset
#random_state = any int value means every time when you run your program you will get the same output for train and test dataset, random_state is None by default which means every time when you run your program you will get different output because of splitting between train and test varies within
test_dataset = dataset.drop(train_dataset.index) #remove train_dataset from dataframe to get test_dataset
train_dataset
train_labels = pd.DataFrame([train_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T
test_labels = pd.DataFrame([test_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T
# Importing Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers
print(tf.__version__)
#!pip install git+https://github.com/tensorflow/docs # Use some functions from tensorflow_docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from keras import regularizers #regularizers performs regularization which is used to reduce Overfitting
model = keras.Sequential([
    layers.Dense(3, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=[len(train_dataset.keys())]), #inputshape=[3]
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(11)
  ])
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])
model.summary()
history = model.fit(x=train_dataset, y=train_labels,
                    validation_split=0.2,
                    epochs=5001,
                    batch_size=2048,
                    verbose=0,
                    callbacks=[tfdocs.modeling.EpochDots()],
                    shuffle=True)
import matplotlib.pyplot as plt
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "accuracy")
plt.ylim([0, 1])
plt.ylabel('accuracy [Color]')
plotter.plot({'Basic': history}, metric = "loss")
plt.ylim([0, 1])
plt.ylabel('loss [Color]')
test_predictions = model.predict(test_dataset)
print("shape is {}".format(test_predictions.shape))
test_predictions
#Selecting Class with highest confidence
predicted_encoded_test_labels = np.argmax(test_predictions, axis=1) #Returns the indices of the maximum values along each row(axis=1)
#Converting numpy array to pandas dataframe
predicted_encoded_test_labels = pd.DataFrame(predicted_encoded_test_labels, columns=['Predicted Labels'])
predicted_encoded_test_labels
#Converting One-Hot Encoded Actual Test set labels into Label Encoding format
actual_encoded_test_labels = np.argmax(test_labels.to_numpy(), axis=1)
#Converting numpy array to pandas dataframe
actual_encoded_test_labels = pd.DataFrame(actual_encoded_test_labels, columns=['Actual Labels'])
actual_encoded_test_labels
model.evaluate(x=test_dataset, y=test_labels)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
confusion_matrix_test = confusion_matrix(actual_encoded_test_labels, predicted_encoded_test_labels)
f,ax = plt.subplots(figsize=(16,12))
categories = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='d',
            xticklabels = categories,
            yticklabels = categories)
plt.show()
model.save('colormodel_trained_90.h5')
model.summary()
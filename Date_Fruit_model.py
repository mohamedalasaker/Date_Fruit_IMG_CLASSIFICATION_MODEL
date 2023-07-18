
#import needed packages

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#load images from google colab disk(in batches)
from zipfile import ZipFile
file_name =  "/content/Date_Fruit_Image_Dataset_Splitted_Train (1).zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()

file_name =  "/content/Date_Fruit_Image_Dataset_Splitted_Test (1).zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/content/train",
  validation_split=0.2,
  seed = 123,
  subset = 'training',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "/content/train",
  validation_split=0.2,
  seed = 123,
  subset = 'validation',
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  "/content/test",
  image_size=(img_height, img_width),
  batch_size=batch_size)



#Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


#Build the CNN model

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.RandomZoom(0.1),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.AveragePooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(9,activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()


#Train the model
epochs=100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  class_weight = {0:0.2,1:1,2:1,3:1.7,4:1,5:1,6:1,7:2,8:1.7},
)


#Make the train and test sets labels as an array
actual_values_train = []
for x,y in train_ds:
  for e in y.numpy():
    actual_values_train.append(e)

actual_values_test = []
for x,y in test_ds:
  for e in y.numpy():
    actual_values_test.append(e)


#Classification report

pred = model.predict(test_ds)
pred = np.argmax(pred, axis = 1)
print(classification_report(actual_values_test, pred,target_names=['Ajwa', 'Galaxy', 'Mejdool', 'Meneifi', 'NabtatAli', 'Rutab', 'Shaishe', 'Sokari', 'Sugaey']))



#cunfiusion matrix
cm = confusion_matrix(actual_values_test,pred);
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Ajwa', 'Galaxy', 'Mejdool', 'Meneifi', 'NabtatAli', 'Rutab', 'Shaishe', 'Sokari', 'Sugaey'])
fig, ax = plt.subplots(figsize=(9,9))
disp.plot(ax=ax) 


#Bar chart for the count of each label in train and test sets


fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(['Ajwa', 'Galaxy', 'Mejdool', 'Meneifi', 'NabtatAli', 'Rutab', 'Shaishe', 'Sokari', 'Sugaey'],np.unique(actual_values_train,return_counts=True)[1],
        width = 0.4)
 
plt.xlabel("Date type")
plt.ylabel("Count")
plt.title("Training dataset")
plt.show()

print("\n\n")
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(['Ajwa', 'Galaxy', 'Mejdool', 'Meneifi', 'NabtatAli', 'Rutab', 'Shaishe', 'Sokari', 'Sugaey'],np.unique(actual_values_test,return_counts=True)[1],
        width = 0.4)
 
plt.xlabel("Date type")
plt.ylabel("Count")
plt.title("Testing dataset")
plt.show()



#Accurcy and loss plots for train and test sets

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(24, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.xlabel("epoch")
plt.ylabel("accurcy")
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#Evaluating the model of test and valdation and train sets

testEval = model.evaluate(test_ds)

print("The test loss: ",testEval[0])
print("The test accuracy: ",testEval[1])

validationEval = model.evaluate(val_ds)

print("The validation loss: ",validationEval[0])
print("The validation accuracy: ",validationEval[1])

trainEval = model.evaluate(train_ds)

print("The train loss: ",trainEval[0])
print("The train accuracy: ",trainEval[1])
#%%
# Building Convolutional Neural Networks to Classify the Dog and Cat Images. This is a Binary Classification Model i.e. 0 or 1
# Used Dataset -- a Subset (10,000) Images ==> (8,000 for training_set: 4,000 Dogs and 4,000 Cats) and (2,000 for test_set: 1,000 Dogs and 1,000 Cats of Original Dataset (25,000 images) of Dogs vs. Cats | Kaggle
# Original Dataset link ==> https://www.kaggle.com/c/dogs-vs-cats/data
# You might use 25 or more epochs and 8000 Samples per epoch

# Installing Theano
# Installing Tensorflow
# Installing Keras

# Part 1 - Building the ConvNet

#%%
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the ConvNet
classifier = Sequential()

# Step 1 - Building the Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Building the Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding The Second Convolutional Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Building the Flattening Layer
classifier.add(Flatten())

# Step 4 - Building the Fully Connected Layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the ConvNet
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the ConvNet to the Images

#%%
from keras.preprocessing.image import ImageDataGenerator
# ..... Fill the Rest (a Few Lines of Code!)

#set batch size
batch_sz = 64


#add Path to parent folder
from pathlib import Path
path_base = Path(__file__).parent.absolute()
print()
print(path_base)
data_path = path_base / "ConvNet_dataset"
print(data_path)

# Create DataGenerator Objects
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# Create generator to provide images
train_generator = train_datagen.flow_from_directory(
        data_path / "training_set",
        target_size=(64, 64),
        batch_size=batch_sz,
        class_mode='binary')
test_generator = test_datagen.flow_from_directory(
         data_path / "test_set",
        target_size=(64, 64),
        batch_size=batch_sz,
        shuffle=False, #Shuffle must be false for second set of predictions
        class_mode='binary')


#Get File names
train_names = train_generator.filenames
test_names = test_generator.filenames

#Set lengths\
train_len = len(train_names)
test_len = len(test_names)

# Fit Model
classifier.fit(
        train_generator,
        steps_per_epoch=train_len/batch_sz,
        epochs=25,
        validation_data=test_generator,
        validation_steps=test_len/batch_sz)


# %%
#Make Predictions
import numpy as np
import pandas as pd

test_generator.reset()
predict_gen = classifier.predict_generator(test_generator,steps = test_len/batch_sz)


predicted_class_indices = []
for prediction in predict_gen:
        predicted_class_indices.append(round(prediction[0]))

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]


results=pd.DataFrame({"Filename":test_names,
                      "Predictions":predictions})

#Shuffle Prediction Results
results=results.sample(frac=1)

#Set Panda to display more rows on print
pd.set_option("display.max_rows", None, "display.max_columns", None)

file_base = []

for filename in results['Filename']:
        file_base.append(filename[:4])
results['file_base'] = file_base

results['correct'] =  np.where(results['file_base'] == results['Predictions'], 1, 0)

print(results.head(50))
#Calculate Metrics
correct = results['correct'].sum()
total = len(results['correct'])
accuracy = correct/total

#print Results
print()
print(f"Number Correct: {correct}")
print(f"Total Predictions: {total}")
print(f"Accuracy: {accuracy}")
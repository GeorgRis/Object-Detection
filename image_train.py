from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and Preprocess Images
train_dir = r'Path_to_train'
test_dir = r'Path_to_test'

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_dir, target_size=(150, 150), color_mode='rgb', batch_size=32, class_mode='binary')
test_generator = datagen.flow_from_directory(test_dir, target_size=(150, 150), color_mode='rgb', batch_size=32, class_mode='binary')

# Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile with adam
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=25, validation_data=test_generator, validation_steps=len(test_generator))

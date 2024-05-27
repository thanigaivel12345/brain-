from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras_tuner import BayesianOptimization
import tensorflow as tf

# Define batch size and number of epochs
batch_size = 32
n_epochs = 30

# Define data generator for training (using the entire dataset)
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Admin\Downloads\BrainHemarage\BrainHemarage',  # Update with the actual path to dataset
    target_size=(200, 200),  # All images will be resized to 200 x 200
    batch_size=batch_size,
    subset='training',  # Use the training subset
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_directory(
    r'C:\Users\Admin\Downloads\BrainHemarage\BrainHemarage',  # Update with the actual path to dataset
    target_size=(200, 200),  # All images will be resized to 200 x 200
    batch_size=batch_size,
    subset='validation',  # Use the validation subset
    class_mode='categorical'
)

# Define the CNN model architecture using Keras Tuner for Bayesian Optimization
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(200, 200, 3)))  # Define input shape explicitly
    model.add(Conv2D(hp.Int('conv1_units', min_value=16, max_value=64, step=16),
                     (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(hp.Int('conv2_units', min_value=16, max_value=64, step=16),
                     (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the Bayesian Optimization tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='bayesian_opt',
    project_name='brain_hemorage'  # Update project name if needed
)

# Perform the hyperparameter search
tuner.search(train_generator, epochs=n_epochs, validation_data=val_generator)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
best_model.save('best_model.h5')

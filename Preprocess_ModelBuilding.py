#preprocessing and building- training CNN
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Set the paths to your training and testing data
training_folder = 'C:/Users/USER/Desktop/BEYOND_WORDS/asl_alphabet_train'
testing_folder = 'C:/Users/USER/Desktop/BEYOND_WORDS/asl_alphabet_test'

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
)

# Use the generators to load images directly from directories
train_generator = train_datagen.flow_from_directory(
    training_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Automatically detects classes from subdirectories
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    training_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

test_generator = test_datagen.flow_from_directory(
    testing_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
)

# Create MobileNetV2 model as the base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Freeze the base model
base_model.trainable = False

# Build the model on top of the base model
model = models.Sequential([
    base_model,
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=None,  # Automatically calculate based on the dataset size
    validation_steps=validation_generator.samples // validation_generator.batch_size,
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc}')

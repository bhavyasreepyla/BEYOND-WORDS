import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize variables
background = None
accumulated_weight = 0.5
ROI_top, ROI_bottom, ROI_right, ROI_left = 100, 300, 150, 350

# Function to calculate accumulated average
def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

# Function to segment the hand
def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)



word_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "G",
             10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
             19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space"}

word_dict_values = list(word_dict.values())




# Initialize the camera
cam = cv2.VideoCapture(0)
num_frames = 0
element = 10
num_imgs_taken = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                        (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elif num_frames <= 300:
        hand = segment_hand(gray_frame)
        cv2.putText(frame_copy, "Adjust hand...Gesture for" + str(element),
                    (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            cv2.putText(frame_copy, str(num_frames) + "For" + str(element),
                        (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Thresholded Hand Image", thresholded)
    else:
        hand = segment_hand(gray_frame)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_copy, str(num_imgs_taken) + 'images' + "For" + str(element),
                        (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Thresholded Hand Image", thresholded)
            if num_imgs_taken <= 300:
                cv2.imwrite("C:/Users/USER/Desktop/BEYOND_WORDS/split_data/train/" + str(element) +
                            str(num_imgs_taken + 300) + '.jpg', thresholded)
            else:
                break
            num_imgs_taken += 1
        else:
            cv2.putText(frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
    num_frames += 1
    cv2.imshow("Sign Detection", frame_copy)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cam.release()

train_path = 'C:/Users/USER/Desktop/BEYOND_WORDS/split_data/train'
test_path = 'C:/Users/USER/Desktop/BEYOND_WORDS/split_data/test'

# Image Data Generator
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_batches = test_datagen.flow_from_directory(test_path, target_size=(64, 64), classes=word_dict_values, batch_size=32)
# Plotting Images

train_batches = train_datagen.flow_from_directory(train_path, target_size=(64, 64), classes=word_dict_values, batch_size=32)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0

        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


imgs, labels = train_batches.next()
plotImages(imgs)

# Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(28, activation="softmax"))  # Change the number of units to match your number of classes

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# Train the model
history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=test_batches, verbose=0)
model.save('train.h5')

# Evaluate the model
scores = model.evaluate(test_batches, steps=len(test_batches), verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

# Make predictions
predictions = model.predict_generator(test_batches, steps=len(test_batches), verbose=0)
print("Predictions on a small set of test data:")
for pred in predictions:
    print(word_dict[np.argmax(pred)], end=' ')

# Load the model for hand gesture prediction
model = keras.models.load_model("train.h5")

max_frames = 10

# Hand gesture prediction
num_frames = 0
while num_frames < max_frames:
    print("Before capturing frame")
    ret, frame = cam.read()
    print("After capturing frame")





    
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 70:
        cal_accum_avg(gray_frame, accumulated_weight)
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                    (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        hand = segment_hand(gray_frame)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            cv2.imshow("Thesholded Hand Image", thresholded)
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))
            pred = model.predict(thresholded)
            #predicted_alphabet = word_dict[np.argmax(pred)]
            predicted_alphabet = word_dict[predicted_class]
            print("Predicted Alphabet:", predicted_alphabet)

            cv2.putText(frame_copy, predicted_alphabet,
                        (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    num_frames += 1
    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _",
                (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
    cv2.imshow("Sign Detection", frame_copy)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()

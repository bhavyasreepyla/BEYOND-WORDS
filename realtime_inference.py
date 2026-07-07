# Real-time ASL alphabet recognition from the webcam.
# Loads the MobileNetV2 model saved by train_model.py and predicts the letter
# shown inside the region of interest (ROI) box.
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "asl_mobilenetv2.keras"

# Dictionary to save our 29 classes
word_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
             10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
             19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space"}

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment_max_cont


def main():
    model = load_model(MODEL_PATH)

    cam = cv2.VideoCapture(0)
    num_frames = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 70:
            cal_accum_avg(gray_frame, accumulated_weight)
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            hand = segment_hand(gray_frame)
            if hand is not None:
                thresholded, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
                cv2.imshow("Thresholded Hand Image", thresholded)

                # The model was trained on 224x224 RGB images with the
                # MobileNetV2 preprocessing, so do the same here.
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_rgb = cv2.resize(roi_rgb, (224, 224))
                roi_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(roi_rgb.astype("float32"))
                roi_rgb = np.expand_dims(roi_rgb, axis=0)

                pred = model.predict(roi_rgb, verbose=0)
                predicted_alphabet = word_dict[int(np.argmax(pred))]

                cv2.putText(frame_copy, predicted_alphabet, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
        num_frames += 1
        cv2.imshow("Sign Detection", frame_copy)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC to quit
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

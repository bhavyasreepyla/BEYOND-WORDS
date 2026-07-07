ASL (American Sign Language) Recognition System

Work in Progress

This project focuses on developing a system for recognizing American Sign Language (ASL) gestures using deep learning techniques. The system employs a Convolutional Neural Network (CNN) for gesture recognition from live video streams captured by a camera. Additionally, it incorporates Natural Language Processing (NLP) to convert recognized gestures into speech, providing real-time audio feedback.

Features

- Gesture Recognitio: Utilizes a trained CNN model to recognize hand gestures corresponding to ASL alphabet letters.
- Text-to-Speech Conversion: Employs NLP techniques to convert recognized gestures into spoken words, aiding communication for individuals with hearing impairments.
- User Interface: Provides an intuitive graphical interface for interacting with the system and visualizing gesture recognition results(using streamlit).

Dataset
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Setup
Prerequisites
- Python 3.x
- Install the dependencies with: pip install -r requirements.txt

Pipeline

1. Download the Kaggle grassknoted/asl-alphabet dataset and extract it into a `data/` folder, so the training images are in `data/asl_alphabet_train/` (and the test images in `data/asl_alphabet_test/`).
2. Run `Extraction_Distribution_Splitting.py` to explore the dataset (sample images, class distribution) and split it into train/val/test folders under `split_data/`.
3. Run `train_model.py` to train the MobileNetV2 classifier. It saves the trained model as `asl_mobilenetv2.keras`.
4. Run `realtime_inference.py` to recognize ASL letters live from your webcam (press ESC to quit).
5. Run `streamlit run text_to_speech_app.py` for the text-to-speech interface.

Usage

1. Launch the application.
2. Position your hand within the camera frame.
3. Perform ASL gestures corresponding to alphabet letters.
4. The system will recognize and convert the gestures into spoken words in real-time.

Future Enhancements

- Model Optimization: Fine-tune the CNN model for improved accuracy and efficiency in gesture recognition.
- Expanded Vocabulary: Include additional ASL gestures and words to enhance communication capabilities.
- User Feedback: Implement audio or visual feedback mechanisms for a better user experience.

Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests with your enhancements or bug fixes.

Acknowledgments

The webcam region-of-interest and background-subtraction hand segmentation approach used in `realtime_inference.py` was adapted from a DataFlair hand sign recognition tutorial.

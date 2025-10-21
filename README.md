MaskID: Face Mask Detection System using CNN

🌟 Overview

MaskID is a deep learning project utilizing a Convolutional Neural Network (CNN) to accurately detect whether individuals in an image are wearing a face mask or not. This system is developed using TensorFlow/Keras and is trained on a public dataset, making it ideal for applications like real-time monitoring of public health compliance in enclosed spaces or public transport.

The project achieves a test accuracy of approximately 92%.

🛠️ Technology Stack

Language: Python

Core Libraries: TensorFlow 2.x, Keras

Data Processing: NumPy, Pandas, OpenCV (cv2), Pillow (PIL)

Data Source: Face Mask Dataset (Kaggle)

📁 Project Workflow

The solution is implemented entirely within the main.ipynb Jupyter Notebook and follows these key steps:

1. Data Acquisition and Preparation

The project uses the Kaggle API to download the face-mask-dataset.zip.

The raw images are organized into two classes: with_mask (3725 images) and without_mask (3828 images).

Labeling: Images with masks are assigned the numerical label 1, and those without a mask are labeled 0.

2. Image Preprocessing

All images undergo uniform preprocessing steps to ensure they are suitable for the CNN model:

Resizing: Images are resized to a fixed resolution of 128x128 pixels.

Format Conversion: Images are converted to RGB format.

Array Conversion: The image data and labels are converted into NumPy arrays (X for features, Y for labels).

Scaling/Normalization: Pixel intensity values (0-255) are scaled down to a range of 0 to 1 by dividing by 255.

3. Model Architecture (Convolutional Neural Network)

A Sequential CNN model is designed for binary image classification:

Layer Type

Filters/Units

Kernel/Pool Size

Activation

Purpose

Conv2D

32

(3, 3)

ReLU

Initial feature extraction

MaxPooling2D

-

(2, 2)

-

Spatial downsampling

Conv2D

64

(3, 3)

ReLU

Deeper feature extraction

MaxPooling2D

-

(2, 2)

-

Spatial downsampling

Flatten

-

-

-

Prepares for dense layers

Dense (Hidden)

128

-

ReLU

Learning complex patterns

Dropout

-

0.5

-

Regularization

Dense (Hidden)

64

-

ReLU

Learning complex patterns

Dropout

-

0.5

-

Regularization

Dense (Output)

2

-

Sigmoid

Final prediction (Mask/No Mask)

4. Training and Evaluation

Training Split: Data is split into 80% for training (X_train, Y_train) and 20% for testing (X_test, Y_test).

Compilation: The model uses the adam optimizer and sparse_categorical_crossentropy loss.

Training: The model was trained for 5 epochs.

Performance:

Final Training Accuracy: ~93.08%

Final Test Accuracy: ~92.19%

🚀 Getting Started

Prerequisites

You need Python 3.x and the following libraries:

pip install tensorflow keras numpy pandas opencv-python pillow scikit-learn kaggle


Running the Notebook

Kaggle Setup: Obtain your kaggle.json credentials file.

Download Data: The notebook handles the data download automatically if your Kaggle API is set up. Otherwise, manually download the "Face Mask Dataset" and extract it to a directory named data/ containing with_mask and without_mask subdirectories.

Execute Cells: Run all cells in main.ipynb sequentially.

Prediction: The final section of the notebook includes a simple predictive system where you can input the path to any image, and the model will output its classification.

# Example of prediction output:
# Path of the image to be predicted: /path/to/your/image.jpg
# [[0.49811754 0.47740024]]
# 0
# The person in the image is not wearing a mask


📈 Model Performance Visualization

The notebook generates plots to show the training and validation loss and accuracy over the epochs, which helps in identifying potential overfitting and model stability.

Developed as a foundational project in CNN and image classification.

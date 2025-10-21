# MaskID: Face Mask Detection System using CNN

This project is a deep learning model built with TensorFlow and Keras to detect whether a person in an image is wearing a face mask. The model is a Convolutional Neural Network (CNN) trained on a public dataset from Kaggle, achieving an accuracy of approximately 92.2% on the test set.

## 📋 Table of Contents
* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Technologies Used](#technologies-used)
* [Setup and Installation](#setup-and-installation)
* [How to Run](#how-to-run)
* [Results](#results)
* [Predictive System](#predictive-system)

## 🚀 Project Overview

"MaskID" is designed to classify images into two categories: 'with_mask' or 'without_mask'. This is particularly relevant for public health and safety monitoring. The system uses a CNN, a class of deep neural networks well-suited for image analysis. The entire pipeline, from data acquisition and preprocessing to model training and prediction, is contained within the `main.ipynb` notebook.

## ✨ Features

* **CNN Model:** Utilizes a sequential CNN model for high-accuracy image classification.
* **Data Augmentation:** Implements image resizing and scaling for uniform data preprocessing.
* **Training & Evaluation:** Includes complete code for training, validation, and testing the model.
* **High Accuracy:** Achieves **92.19%** accuracy on the unseen test dataset.
* **Predictive System:** A simple, interactive prediction script is included to test the model on new images.

## 📊 Dataset

The model is trained on the **Face Mask Dataset** available on Kaggle, which was contributed by Omkar Gurav.
* **Dataset Link:** [https://www.kaggle.com/datasets/omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
* The dataset contains 7,553 images, split into two classes:
    * `with_mask`: 3,725 images (Labelled as **1**)
    * `without_mask`: 3,828 images (Labelled as **0**)

All images are preprocessed by resizing them to `128x128` pixels and converting them to RGB.

## 🧠 Model Architecture

The model is a `keras.Sequential` CNN with the following layers:

1.  **Conv2D:** 32 filters, (3,3) kernel, 'relu' activation, input shape (128, 128, 3)
2.  **MaxPooling2D:** (2,2) pool size
3.  **Conv2D:** 64 filters, (3,3) kernel, 'relu' activation
4.  **MaxPooling2D:** (2,2) pool size
5.  **Flatten**
6.  **Dense:** 128 units, 'relu' activation
7.  **Dropout:** 0.5
8.  **Dense:** 64 units, 'relu' activation
9.  **Dropout:** 0.5
10. **Output Layer (Dense):** 2 units, 'sigmoid' activation

The model is compiled using:
* **Optimizer:** `adam`
* **Loss Function:** `sparse_categorical_crossentropy`
* **Metrics:** `acc`

## 🛠️ Technologies Used

* Python 3
* TensorFlow & Keras
* NumPy
* OpenCV (`cv2`)
* Pillow (`PIL`)
* Scikit-learn (`sklearn`)
* Matplotlib
* Kaggle API

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/MaskID.git](https://github.com/your-username/MaskID.git)
    cd MaskID
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib opencv-python pillow scikit-learn kaggle
    ```

3.  **Set up Kaggle API:**
    * Go to your Kaggle account, click on your profile picture, and select "Account".
    * Click "Create New API Token" to download `kaggle.json`.
    * Place the `kaggle.json` file in the root directory of this project.
    * The notebook will automatically create the `~/.kaggle/` directory and set the correct permissions.

## 🚀 How to Run

1.  Open the `main.ipynb` file in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab).
2.  Run the cells sequentially.
3.  The notebook will:
    * Install and configure the Kaggle API.
    * Download and extract the dataset.
    * Load, preprocess, and label the images.
    * Split the data into training (80%) and testing (20%) sets.
    * Build the CNN model.
    * Train the model for 5 epochs.
    * Evaluate the model on the test set and display the results.

## 📈 Results

The model was trained for 5 epochs with a validation split of 10%.
* **Final Test Accuracy:** **92.19%**

### Training History

The plots below show the model's loss and accuracy over the 5 epochs of training.

**Model Loss**
*(The training loss decreases steadily, while the validation loss shows good generalization.)*

**Model Accuracy**
*(The training and validation accuracy both increase, indicating the model is learning effectively.)*

## 🔮 Predictive System

The notebook includes a final cell to test the trained model on any image.

1.  Run the final cell.
2.  You will be prompted to enter the path of an image.
3.  The model will load the image, process it, and print its prediction.

**Example 1 (With Mask):**

* **Input:** `/content/test.png`
* **Output:** `The person in the image is wearing a mask`

**Example 2 (Without Mask):**

* **Input:** `/content/test.jpg`
* **Output:** `The person in the image is not wearing a mask`

---
*This README was generated for the MaskID project.*

# **Title: Face Mask Detection System**


## **1. Methodology**
<img src="https://user-images.githubusercontent.com/7460892/207003643-e03c8964-3f16-4a62-9a2d-b1eec5d8691f.png" width="80%" height="80%">

## **2. Description**
<img src="<img width="350" height="118" alt="Screenshot 2025-11-12 175605" src="https://github.com/user-attachments/assets/b77d0774-45ab-413c-9e2a-5927d626884f" />
">



## ✨ Features

* **Binary Image Classification:** Classifies images into 'with_mask' or 'without_mask'.
* **Deep Learning Model:** Uses a sequential Convolutional Neural Network (CNN).
* **High Accuracy:** Achieves **~92.2%** accuracy on the test dataset.
* **Data Preprocessing:** Includes a full pipeline for loading, resizing (128x128), and normalizing images.
* **Predictive System:** A simple, interactive script is included to test the model on new, unseen images.

## 📊 Dataset

The model is trained on the **Face Mask Dataset** available on Kaggle, which was contributed by Omkar Gurav.

* **Dataset Link:** [https://www.kaggle.com/datasets/omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
* **Total Images:** 7,553
* **Classes:**
    * `with_mask`: 3,725 images (Labelled as **1**)
    * `without_mask`: 3,828 images (Labelled as **0**)

## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow** & **Keras** (for building and training the CNN)
* **Scikit-learn** (for splitting the data)
* **OpenCV (`cv2`)** & **Pillow (`PIL`)** (for image loading and processing)
* **NumPy** (for array manipulation)
* **Matplotlib** (for plotting results)
* **Kaggle API** (for downloading the dataset)

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
    * Go to your Kaggle account page and click "Create New API Token" to download `kaggle.json`.
    * Place the `kaggle.json` file in the same directory as the `main.ipynb` notebook.
    * The notebook includes commands to automatically move this file to the correct location (`~/.kaggle/`) and set its permissions.

## 🚀 How to Run

1.  Open the `main.ipynb` file in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab).
2.  Run the cells sequentially. The notebook will:
    * Install and configure the Kaggle API.
    * Download and extract the dataset.
    * Load, preprocess (resize to 128x128, convert to 'RGB'), and label all images.
    * Split the data into training (80%) and testing (20%) sets.
    * Scale the pixel values of the images (dividing by 255).
    * Build the CNN model.
    * Compile and train the model for 5 epochs.
    * Evaluate the model on the test set and print the final accuracy.
    * Plot the training/validation loss and accuracy.


## 📈 Results

The model was trained for 5 epochs and achieved the following result on the held-out test set:

* **Test Accuracy: 92.19%**

## 🔮 Predictive System

The notebook includes a final cell to test the trained model on any local image.

1.  Run the final cell.
2.  You will be prompted to enter the path of an image.
3.  The model will load the image, process it, and print the model's prediction.

## 🏁 Conclusion

Building this project was a valuable exercise in applying deep learning to a tangible, socially relevant problem. While the immediate crisis has subsided, the principles behind this system are still valuable for automated monitoring in various public health and safety contexts.

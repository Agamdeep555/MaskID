# MaskID: Face Mask Detection System using CNN

## 🚀 Project Overview & Motivation

This project, **MaskID**, was born from a reflection on the global pandemic that began over four years ago. It struck me how a simple piece of cloth—a face mask—became one of the most critical tools in our collective fight against a global health crisis.

During that time, ensuring public safety in spaces like airports, hospitals, and essential businesses was paramount. Manual monitoring of mask compliance was difficult, inefficient, and often put staff at risk. This sparked the idea for the project: a simple, effective, and automated system to detect face mask usage in real-time.

This repository is my exploration of building such a system. It serves as a technical case study on how Convolutional Neural Networks (CNNs) could be rapidly deployed to solve an immediate, critical, real-world problem. It’s a look back at the kind of technology that was essential for safeguarding public health during one of the most challenging periods of our time.

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

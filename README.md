# **Title: Face Mask Detection System**


## 1. Methodology
<img src="https://user-images.githubusercontent.com/7460892/207003643-e03c8964-3f16-4a62-9a2d-b1eec5d8691f.png" width="80%" height="80%">

## 2. Description
<img width="500" height="200" alt="Screenshot 2025-11-12 175605" src="https://github.com/user-attachments/assets/dffa3ebe-fc38-4a60-b8b9-42052314ac0a" />



## 3. Dataset

The model is trained on the **Face Mask Dataset** available on Kaggle, which was contributed by Omkar Gurav.

* **Dataset Link:** [https://www.kaggle.com/datasets/omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
* **Total Images:** 7,553
* **Classes:**
    * `with_mask`: 3,725 images (Labelled as **1**)
    * `without_mask`: 3,828 images (Labelled as **0**)

## 4. Technologies Used

* **Python 3**
* **TensorFlow** & **Keras** (for building and training the CNN)
* **Scikit-learn** (for splitting the data)
* **OpenCV (`cv2`)** & **Pillow (`PIL`)** (for image loading and processing)
* **NumPy** (for array manipulation)
* **Matplotlib** (for plotting results)
* **Kaggle API** (for downloading the dataset)


## 5. Results

The model was trained for 5 epochs and achieved the following result on the held-out test set:

* **Test Accuracy: 92.19%**


## 6. Conclusion

Building this project was a valuable exercise in applying deep learning to a tangible, socially relevant problem. While the immediate crisis has subsided, the principles behind this system are still valuable for automated monitoring in various public health and safety contexts.

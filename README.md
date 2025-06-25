# 🐶 Multi-Class Dog Breed Classification with TensorFlow

This project builds a **deep learning image classifier** that predicts the breed of a dog from an input photo using **transfer learning** and **TensorFlow Hub**. It is based on the Kaggle Dog Breed Identification dataset and implements a complete pipeline from data preprocessing to model evaluation.

---

## 🧠 Project Overview

The goal is to classify an image of a dog into one of **120 possible breeds**. This type of tool could be useful in public or private applications such as:
- Dog breed recognition apps
- Veterinary screening support tools
- Rescue shelters' intake systems

The final model uses **Google’s MobileNetV2**, a lightweight pre-trained CNN architecture available from TensorFlow Hub, fine-tuned on the dog breed dataset.

---

## 📂 Dataset

- **Source**: [Kaggle Dog Breed Identification Competition](https://www.kaggle.com/c/dog-breed-identification)
- **Training set**: ~10,000 labeled images
- **Test set**: ~10,000 unlabeled images
- **Total classes**: 120 distinct dog breeds
- **Data type**: Unstructured image data (JPEG)

---

## 🎯 Problem Statement

Develop a multi-class classifier to:
- Predict the **probability distribution** across all 120 breeds for each image
- Minimize **multi-class log loss**, the competition's evaluation metric

---

## 🛠️ Tools & Technologies

- Python 3.11.11  
- TensorFlow 2.15  
- TensorFlow Hub 0.15  
- TensorFlow-Metal 1.1.0 *(for Mac GPU acceleration)*  
- Jupyter Notebook  
- Scikit-learn 1.3.2  
- Pandas 2.2.1  
- NumPy *(compatible with TF 2.15)*  
- Matplotlib 3.8.3  

---

## 🔍 Notebook Structure

### 1. Project Setup
- Problem definition  
- Data overview  
- Evaluation metric (multi-class log loss)  
- Feature summary  

### 2. Data Preparation
- Workspace preparation  
- Getting data ready (turning images into tensors)  
- Getting image file paths and labels  
- Creating a custom validation set  
- Preprocessing and batching the data  
- Visualizing image batches  

### 3. Model Development
- Building a model using MobileNetV2 (via TensorFlow Hub)  
- Creating callbacks  
  - TensorBoard callback  
  - Early stopping callback  

### 4. Training and Evaluation
- Training on a subset of 1,000 images (pipeline sanity check)  
- Visualizing training metrics with TensorBoard  
- Making predictions and evaluating performance  
- Saving and reloading the trained model  

### 5. Full Dataset Training
- Training the model on the complete dataset  
- Making predictions on the test set  
- Preparing a `.csv` file for Kaggle submission  

### 6. Bonus
- Making predictions on custom (user-supplied) images  

---

## 📈 Results

- **Evaluation metric**: Multi-class log loss  
- **Output**: `.csv` file with prediction probabilities for each test image (for Kaggle submission)  
- The model demonstrates reasonable accuracy and generalization on unseen dog images

---

## 🚀 Future Improvements

- 🔧 **Hyperparameter Tuning**: Experiment with different learning rates, optimizers, regularization strategies  
- 🧠 **Layer Unfreezing**: Fine-tune deeper layers of MobileNetV2 after initial training  
- 🖼️ **Data Augmentation**: Use geometric transforms, color jitter, and scaling for better generalization  
- 📈 **Dataset Expansion**: Increase image quantity and improve label quality  

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Kaggle for dataset and competition platform  
- TensorFlow & TensorFlow Hub for pre-trained model architecture  
- The open-source community for tutorials and inspiration
- The Zero To Mastery Academy for their teachings and mentorship

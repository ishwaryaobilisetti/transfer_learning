# Dogs vs Cats Image Classification using Transfer Learning

## Objective
The objective of this project is to build a high-performance image classifier for a custom dataset using transfer learning. A pre-trained convolutional neural network is adapted to classify images of dogs and cats efficiently, following modern deep learning best practices.

## Dataset
This project uses the **Cats vs Dogs** dataset provided by **TensorFlow Datasets (TFDS)**. The dataset is publicly available and accessible to all users without requiring competition access or private links, ensuring full reproducibility.

- Dataset name: cats_vs_dogs
- Classes: Dogs, Cats
- Total images: Approximately 23,000
- Dataset split:
  - Training set: 70%
  - Validation set: 15%
  - Test set: 15%

## Data Preprocessing
All images are resized to 224 × 224 pixels and normalized to a [0,1] range. The dataset is converted into TensorFlow datasets suitable for efficient GPU training.

## Data Augmentation
To improve generalization and reduce overfitting, data augmentation is applied only to the training dataset:
- Random horizontal flip
- Random rotation
- Random zoom

Validation and test datasets are not augmented to ensure unbiased evaluation.

## Model Architecture
The model uses **ResNet50**, pre-trained on ImageNet, as the base model.

- Pre-trained model: ResNet50
- Weights: ImageNet
- Top layers removed
- Base model initially frozen

Custom classification head:
- Global Average Pooling layer
- Dense layer with 256 units and ReLU activation
- Dropout layer with rate 0.5
- Output layer with sigmoid activation for binary classification

## Training Strategy
Training is performed in two phases.

### Phase 1: Feature Extraction
- The ResNet50 base model is frozen
- Only the custom classification head is trained
- Optimizer: Adam
- Learning rate: 1e-3

### Phase 2: Fine Tuning
- Top layers of the ResNet50 base model are unfrozen
- The entire model is trained end-to-end
- Learning rate reduced to 1e-5 to prevent catastrophic forgetting

## Model Evaluation
The final model is evaluated on the test dataset using:
- Accuracy
- Precision
- Recall
- F1-score

A confusion matrix is generated to analyze misclassification patterns between dogs and cats.

## Model Interpretability
Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize which regions of the input images influence the model’s predictions. The heatmaps confirm that the model focuses on meaningful features such as animal faces and body regions.

## Baseline Model Comparison
A simple convolutional neural network trained from scratch is used as a baseline. The transfer learning model significantly outperforms the baseline model in both accuracy and convergence speed.

## Saved Model
The final trained model is saved as:
cats_vs_dogs_resnet50.h5

## Project Files
- transfer.ipynb – Complete executable notebook
- cats_vs_dogs_resnet50.h5 – Final trained model
- requirements.txt – Dependency list
- README.md – Project documentation

## Execution Instructions
1. Clone the repository
2. Install dependencies using requirements.txt
3. Open transfer.ipynb
4. Run all cells sequentially without modification

## Environment
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

The project was developed and executed using Google Colab with GPU acceleration.

## Conclusion
This project demonstrates an end-to-end transfer learning workflow including data preprocessing, two-phase training, evaluation, and interpretability. The implementation is reproducible, well-documented, and satisfies all project submission requirements.

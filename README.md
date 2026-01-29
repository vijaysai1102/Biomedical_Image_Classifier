## model weights

Due to GitHub size limits, trained model files are not included.

Download models here:
- CNN model: https://drive.google.com/file/d/1Y39duosMWga5rfD8Jmiu4ezA0hqzLjxJ/view?usp=sharing
- ResNet50 model: https://drive.google.com/file/d/1usp46btVZFXR4srEiHCJYQ2v9rn9_k3d/view?usp=sharing


## README FOR: project_ResNet50_finetunedu.ipynb##

Purpose:
This notebook implements PneumoNet, a deep learning model for pneumonia detection using transfer learning with ResNet50 pretrained on ImageNet. It improves on the baseline CNN and adds explainability and scalability.

Dependencies:
torch, torchvision, scikit-learn, matplotlib, seaborn, gradio, pyspark, google.colab

Cell 1 – Import Libraries
Imports all libraries required for data processing, deep learning, visualization, and explainability.
Includes torch, torchvision, sklearn.metrics, numpy, seaborn, matplotlib, gradio, pyspark, google.colab.

Cell 2 – Mount Google Drive
Mounts Google Drive to access the chest X-ray dataset stored at /content/drive/MyDrive/chest_xray.

Cell 3 – Install and Configure PySpark
Installs Java (openjdk-11) and pyspark version 3.5.1.
Sets JAVA_HOME and PATH environment variables.
Creates a SparkSession named “PneumoNet_Preprocessing”.
Purpose: to enable distributed preprocessing of image data.

Cell 4 – Load Dataset with Spark
Uses Spark’s image format to read and verify images from dataset_path.
Can detect corrupt files or missing labels.
Caches dataset in memory for faster access.

Cell 5 – Define Torch Transformations, Create Datasets and DataLoaders
Specifies torchvision transforms for preprocessing: Resize(224x224), RandomRotation, RandomHorizontalFlip, RandomResizedCrop, ColorJitter, and Normalize using ImageNet mean and std.
Applies different transformations for training, validation, and test sets.

Uses torchvision.datasets.ImageFolder to create training, validation, and test datasets.
Wraps them with DataLoader using batch_size=32 and num_workers=0 to prevent multiprocessing hang in Colab.

Cell 6 – Load Pretrained ResNet50 Model
Loads ResNet50 pretrained on ImageNet from torchvision.models.
Replaces the final fully connected layer (2048 → 2) for binary classification.
Initially freezes all layers except the final classifier head.

Cell 7 – Define Loss, Optimizer, and Scheduler
Defines CrossEntropyLoss, Adam optimizer with lr=0.0001, and StepLR scheduler (step=3, gamma=0.1).
Sets up model for first training stage (frozen layers).

Cell 8 – Training Stage 1 (Frozen Layers)
Trains the model for 12 epochs with all convolutional layers frozen.
Tracks and prints train and validation loss and accuracy per epoch.
Saves best model weights to /content/drive/MyDrive/pneumonet_resnet50_best.pth.

Cell 9 – Unfreeze, Fine-Tune All Layers and then saving the model
Unfreezes all layers using param.requires_grad = True.
Reinitializes optimizer with smaller learning rate (1e-5).
Fine-tunes model for 5 additional epochs.
Prints performance after each epoch.

cell 10 - Loading the saved model

Cell 11 – Evaluate on Test Dataset
Evaluates the fine-tuned model on the held-out test set.
Computes accuracy, precision, recall, F1-score, and ROC-AUC.
Generates confusion matrix and ROC curve for visualization.

Cell 12-15 – Plot Training and Validation Curves
Plots training vs validation accuracy and loss curves for both training stages to show convergence and overfitting behavior.

Cell 16 – Explainability (Grad-CAM)
Applies Grad-CAM to visualize which lung regions influenced predictions.

Cell 17 – Gradio Deployment
Creates a Gradio interface to upload chest X-ray images and obtain PneumoNet predictions in real time.
Displays model confidence and Grad-CAM heatmap overlay for transparency.

cell 18 - Dependencies 
use this to install all the dependencies - !pip install shap lime scikit-image

cell 19 - Importing necessary libraries

cell 20 - Taking images for the SHAP and LIME explainability

cell 21 and 22 - Explainability (SHAP, LIME)
Uses SHAP and LIME for pixel-level explanation of classification decisions.
Requires torchcam, shap, and lime libraries if available.


Expected Outcome:
Validation accuracy up to 100%, test accuracy around 92–93%, precision 90.5%, recall 98.2%, F1-score 94.2%, and ROC-AUC around 97.5%.
ResNet50 clearly outperforms the baseline CNN with smoother convergence and stronger generalization.


## README FOR: project_baseline_new.ipynb##

Purpose:
This notebook implements a custom Convolutional Neural Network (CNN) from scratch for pneumonia detection using chest X-ray images. It provides a baseline model to compare against advanced transfer learning models like ResNet50. It also introduces initial explainability using SHAP and LIME.

Dependencies:
torch, torchvision, torchaudio, scikit-learn, numpy, matplotlib, seaborn, gradio, shap, lime

Cell 1 – Import Libraries
Imports all necessary Python libraries for deep learning, visualization, and interpretability.
Includes torch, torchvision, sklearn.metrics, numpy, matplotlib, seaborn, gradio, shap, and lime.

Cell 2 – Mount Google Drive
Mounts Google Drive to access the dataset folder /content/drive/MyDrive/chest_xray.
Dependency: google.colab

Cell 3 – Define Dataset Paths and Image Transformations
Defines paths for train, validation, and test subdirectories.
Sets up image preprocessing using torchvision transforms: Resize(224x224), ToTensor, and Normalize(mean, std).

Cell 4 – Load Dataset and Create DataLoaders
Loads train, validation, and test datasets using torchvision.datasets.ImageFolder.
Creates DataLoaders for each with batch_size=32 and shuffle=True for training.
Ensures reproducibility and efficient data feeding to the model.

Cell 5 – Define CNN Architecture
Implements a 3-layer convolutional neural network:
Conv → ReLU → MaxPool repeated thrice, followed by Flatten → Fully Connected (256 neurons, ReLU, Dropout=0.4) → Output layer (2 neurons, softmax).
This simple architecture provides a performance baseline.

Cell 6 – Initialize Model, Loss Function, and Optimizer
Moves model to GPU if available.
Defines loss as CrossEntropyLoss and optimizer as Adam(lr=0.001).
Sets number of epochs (10) and batch size (32).

Cell 7 – Training Loop
Trains the CNN over 10 epochs:

Switches between train and validation mode each epoch
Computes average loss and accuracy
Displays results each epoch

Saves the best model checkpoint to Google Drive or local storage
Dependencies: torch.no_grad, torch.save

Cell 8 – Model Evaluation on Test Set
Evaluates the trained CNN on the test dataset.
Calculates Accuracy, Precision, Recall, F1-score, and ROC-AUC.
Prints confusion matrix and summary metrics for performance reporting.

Cell 9 – Plot Training and Validation Curves
Plots accuracy and loss vs epoch for both training and validation sets using matplotlib.
Used to visualize convergence and detect overfitting.

Cell 10 – Confusion Matrix Visualization
Generates confusion matrix using sklearn.metrics.confusion_matrix and seaborn heatmap.
Displays the number of correctly and incorrectly classified images.

Cell 11 – Grad-CAM Visualization
Uses Grad-CAM to visualize which parts of the image the CNN focused on while predicting Pneumonia or Normal.
Displays heatmaps overlayed on X-ray images.
Dependencies: torchcam or custom gradient-based code.

Cell 12 – SHAP Explainability
Applies SHAP (SHapley Additive exPlanations) to interpret model predictions.
Shows which pixel regions contribute positively or negatively to the Pneumonia class probability.
Outputs summary plots and example overlays highlighting key image areas.
Dependencies: shap

Purpose: to understand how the CNN makes decisions at the feature level.

Cell 13 – LIME Explainability
Uses LIME (Local Interpretable Model-Agnostic Explanations) for pixel-level local interpretability.
Generates superpixel-based visualization explaining why a specific prediction was made for one image.
Dependencies: lime

Purpose: to validate if model reasoning matches clinical relevance (e.g., focuses on lungs, not edges).

Cell 14 – Gradio Web Interface
Creates an interactive web app to test the trained CNN.
Users can upload a chest X-ray image, receive prediction (Normal or Pneumonia), and view model confidence percentage.
Dependencies: gradio, torch, torchvision.transforms

Expected Output:
Train accuracy: ~97%
Validation accuracy: ~62–65%
Test accuracy: ~74–75%
SHAP and LIME highlight that the baseline CNN often focuses on limited or non-critical areas, explaining the overfitting pattern and motivating ResNet50 improvements.



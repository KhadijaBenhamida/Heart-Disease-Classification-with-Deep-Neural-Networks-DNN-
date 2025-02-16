# Heart Disease Prediction

## Description
This project aims to predict the likelihood of heart disease based on various medical features such as age, cholesterol levels, blood pressure, and maximum heart rate. The dataset consists of features that are typically used by doctors to diagnose heart disease. The goal is to evaluate the effectiveness of different machine learning models for binary classification, where the target variable indicates whether the patient has heart disease (1) or not (0).

### The project uses the following models:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Deep Neural Network (DNN)**

The models are evaluated based on accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset
The dataset used in this project contains health-related features of patients, along with a target variable that indicates whether the patient has heart disease (1) or not (0). 

The dataset is split into two files:
- `heart_disease_train.csv`: The training dataset containing the features and target variable.
- `heart_disease_test.csv`: The testing dataset, which is used to evaluate the models.

The features in the dataset include:
- `age`: Age of the patient
- `resting.bp.s`: Resting blood pressure
- `cholesterol`: Cholesterol level
- `max.heart.rate`: Maximum heart rate achieved
- `oldpeak`: Depression induced by exercise relative to rest

## File Structure
```
/content
    /heart_disease
        - heart_disease_train.csv
        - heart_disease_test.csv
    heart-disease-predictor-xm.zip
```

## Requirements
Before running this project, make sure to install the following Python libraries:

- `pandas` (for data manipulation)
- `scikit-learn` (for machine learning algorithms)
- `tensorflow` (for deep learning)
- `matplotlib` (for visualizations)
- `google-colab` (for mounting Google Drive and file access)

You can install these dependencies using pip:

```bash
pip install pandas scikit-learn tensorflow matplotlib google-colab
```

## How to Run the Code

### Step 1: Clone the Repository

Clone this repository to your local machine or directly use Google Colab.

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
```

### Step 2: Navigate to the Project Directory

```bash
cd heart-disease-prediction
```

### Step 3: Download and Unzip the Dataset

Make sure to download the dataset (`heart-disease-predictor-xm.zip`) and extract it into the `/content/heart_disease` folder.

### Step 4: Run the Python Script

Run the script by executing the following command:

```bash
python heart_disease_prediction.py
```

This script will process the dataset, train the models, and display the evaluation metrics.

### Step 5: View the Results

After running the script, the results for each model will be displayed in the terminal, including the following:
- Model accuracy
- Classification report
- Confusion matrix
- Visualizations for DNN accuracy and loss

## Model Evaluation

The models used in this project are:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Deep Neural Network (DNN)**

Each model is evaluated based on its **accuracy**, **precision**, **recall**, and **F1-score**. The best performing model is identified based on these metrics.

### Example Output:

```bash
Decision Tree Accuracy: 0.85
Random Forest Accuracy: 0.89
Logistic Regression Accuracy: 0.86
KNN Accuracy: 0.83
SVM Accuracy: 0.84
DNN Accuracy: 0.91
```

## Visualizations

**Training and Validation Accuracy** and **Training and Validation Loss** for the Deep Neural Network (DNN) model are plotted using `matplotlib`. These graphs help in understanding the model's learning curve and performance over epochs.

```python
import matplotlib.pyplot as plt

# Plotting training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# Plotting training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()
```

## Results Summary

A summary of the accuracies of all the models used is presented in the table below:

| Model                  | Accuracy  |
|------------------------|-----------|
| Decision Tree          | 0.85      |
| Random Forest          | 0.89      |
| Logistic Regression    | 0.86      |
| K-Nearest Neighbors    | 0.83      |
| Support Vector Machine | 0.84      |
| Deep Neural Network    | 0.91      |

![Model Overview](https://github.com/KhadijaBenhamida/Heart-Disease-Classification-with-Deep-Neural-Networks-DNN-/blob/main/img_dnn_heart_disease_prjct.png)

## Conclusion

The Deep Neural Network (DNN) achieved the best performance with an accuracy of **91%**, outperforming traditional machine learning models such as Decision Tree, Random Forest, and Logistic Regression.

## Contributing

Feel free to fork the repository, create branches, and submit pull requests for new features or bug fixes. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is sourced from [source/link].
- This project utilizes machine learning algorithms from `scikit-learn` and deep learning models from `tensorflow`.
- Thanks to the contributors of these libraries for making machine learning and deep learning accessible.

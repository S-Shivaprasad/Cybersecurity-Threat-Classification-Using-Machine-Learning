# Cybersecurity Threat Classification Using Machine Learning

## 📌 Project Overview
This project focuses on classifying cybersecurity threats using machine learning techniques. The dataset used is CICIDS2017, which contains network traffic data labeled with attack categories. The goal is to preprocess the data, train ML models, and evaluate their performance.

## 📂 Dataset
- **Source**: CICIDS2017 Dataset ([Link](https://www.unb.ca/cic/datasets/ids.html))
- **Files Used**: Network traffic logs containing normal and attack traffic data.
- **Attributes**: Contains flow-based network traffic features, including timestamps, source/destination IPs, and protocol information.

## ⚙️ Steps Followed
### 1️⃣ Data Preprocessing
- Merged multiple CSV files into a single dataset.
- Handled missing, infinite, and categorical values.
- Standardized column names and normalized data.
- Feature selection based on correlation analysis.
- Converted categorical labels into numerical format for model compatibility.

### 2️⃣ Model Selection & Training
Two machine learning models were trained:
- **Random Forest Classifier** (Traditional ML Approach)
- **Neural Network (Deep Learning Model)** (Advanced AI Approach)

**Training Strategy:**
- Split the dataset into **80% training** and **20% testing**.
- Used **MinMaxScaler** for feature scaling.
- Applied **GridSearchCV** for hyperparameter tuning in Random Forest.
- Implemented a **Multi-Layer Perceptron (MLP)** neural network with multiple dense layers.

### 3️⃣ Evaluation Metrics
- **Accuracy** - Measures overall correctness.
- **Precision** - Measures the fraction of true positives among predicted positives.
- **Recall** - Measures the fraction of actual positives correctly identified.
- **F1-score** - Harmonic mean of precision and recall.
- **Confusion Matrix** - Visual representation of classification performance.

## 📊 Results & Comparisons
### **Random Forest Performance:**
- **Accuracy**: 99.94%
- **Precision**: 99.94%
- **Recall**: 99.94%
- **F1-score**: 99.94%

### **Neural Network Performance:**
- **Accuracy**: 77.19%
- **Precision**: 69.38%
- **Recall**: 77.19%
- **F1-score**: 67.82%

### 🔄 **Key Observations**
- **Random Forest** performed exceptionally well with high accuracy and F1-score.
- **Neural Network** underperformed, possibly due to:
  - Insufficient hyperparameter tuning.
  - Imbalanced dataset affecting training.
  - Need for additional feature engineering.
- **Further improvements** can be made using:
  - **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
  - **Feature engineering** to refine model inputs.
  - **More advanced architectures** for the neural network (e.g., CNN or RNN for sequential data).

## 🚀 How to Run
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Jupyter Notebook
```bash
jupyter notebook ML_Engineer_Assignmant.ipynb
```

### 3️⃣ (Optional) Train Neural Network Model
```python
python train_neural_network.py
```

## 📎 Files in This Repository
- `ML_Engineer_Assignmant.ipynb` - Jupyter Notebook with full implementation
- `Cybersecurity_Threat_Classification_Report.pdf` - Summary report
- `train_neural_network.py` - Script for training the deep learning model
- `requirements.txt` - List of required dependencies
- `README.md` - Project documentation

## 📌 Contact
For queries, reach out via email at **shivaprasad20005@gmail.com**.

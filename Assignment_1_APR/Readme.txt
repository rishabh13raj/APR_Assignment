Assignment-1: Logistic Regression on Titanic Dataset

Objective
The objective of this assignment is to apply the **Logistic Regression** algorithm to a real-world dataset (Titanic) and evaluate its performance using various metrics. This assignment demonstrates understanding of several pattern recognition concepts, including:
- Data preprocessing
- Binary classification
- Model evaluation
- Probability and feature importance
- Bias-variance tradeoff (brief discussion)

---

Dataset
Dataset Name: Titanic Dataset  
**Source**: [GitHub - DataScienceDojo](https://github.com/datasciencedojo/datasets)  
**Direct Link**: [https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

### Features:
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Male or Female
- `Age`: Age of passenger
- `SibSp`: # of siblings / spouses aboard
- `Parch`: # of parents / children aboard
- `Fare`: Fare paid
- `Embarked`: Port of Embarkation (C/Q/S)

### Target:
- `Survived`: Whether passenger survived (1) or not (0)

---

## ⚙️ Algorithm Used

**Logistic Regression** from `scikit-learn` was used for binary classification.  
Model performance was evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  
- Feature Importance (based on model coefficients)

---

## 🧪 Project Structure


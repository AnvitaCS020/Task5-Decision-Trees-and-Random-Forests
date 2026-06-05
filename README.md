# Decision Trees and Random Forests – Heart Disease Prediction

## Overview

This project demonstrates the implementation of **Decision Tree** and **Random Forest** classifiers using the **Heart Disease dataset**. The objective is to predict whether a patient has heart disease and compare the performance of these two powerful tree-based machine learning models.

---

## Objectives

* Train a **Decision Tree Classifier**
* Visualize the Decision Tree structure
* Analyze and control **overfitting** using tree depth
* Train a **Random Forest Classifier**
* Compare performance of both models
* Interpret **feature importance**
* Evaluate models using **Cross-Validation**

---

## Dataset

The dataset used is **heart.csv**.

### Target Variable:

* `0` → No Heart Disease
* `1` → Heart Disease

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Graphviz

---

## Project Structure

```
ml-task5/
│
├── task5.py
├── heart.csv
├── decision_tree.png
├── feature_importance.png
└── README.md
```

---

## Installation

Install the required dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn graphviz
```

---

## Run the Project

Execute the script using:

```bash
python task5.py
```

---

## Implementation

### Decision Tree

* Trained a **Decision Tree Classifier**
* Visualized the tree using Graphviz
* Controlled overfitting using `max_depth`

### Random Forest

* Trained a **Random Forest Classifier**
* Compared accuracy with Decision Tree
* Extracted and visualized **feature importance**

---

## Model Evaluation

The models were evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix
* 5-Fold Cross Validation

---

## Output Files

| File Name              | Description                           |
| ---------------------- | ------------------------------------- |
| decision_tree.png      | Visualization of the Decision Tree    |
| feature_importance.png | Feature importance from Random Forest |

---

## Key Learnings

* Understanding **Decision Tree algorithms**
* Preventing **overfitting** using depth control
* Using **ensemble learning** with Random Forests
* Interpreting **feature importance**
* Evaluating models using **Cross-Validation**

---

## Conclusion

The **Random Forest model** generally provides better accuracy and generalization compared to a single Decision Tree. By combining predictions from multiple trees, it reduces overfitting and improves performance.

This project highlights the effectiveness of **tree-based machine learning algorithms** for classification problems like heart disease prediction.

---

## Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Use of additional datasets
* Deployment using Flask/Streamlit
* Comparison with other ML models (SVM, Logistic Regression)

---

## 👩‍💻 Author

Anvita R

---

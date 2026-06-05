Task 5: Decision Trees and Random Forests
Overview
This project demonstrates the implementation of Decision Tree and Random Forest classifiers using the Heart Disease dataset. The goal is to predict whether a patient has heart disease and compare the performance of both tree-based machine learning models.
Objectives
•	Train a Decision Tree Classifier
•	Visualize the Decision Tree
•	Analyze and control overfitting using tree depth
•	Train a Random Forest Classifier
•	Compare model performance
•	Interpret feature importance
•	Evaluate models using Cross-Validation
Dataset
The project uses the Heart Disease Dataset (heart.csv).
Target Variable:
•	0 → No Heart Disease
•	1 → Heart Disease
Technologies Used
•	Python
•	Pandas
•	NumPy
•	Matplotlib
•	Scikit-learn
•	Graphviz
Project Structure
ml-task5/
│
├── task5.py
├── heart.csv
├── decision_tree.png
├── feature_importance.png
└── README.md
Installation
Install the required packages:
pip install pandas numpy matplotlib scikit-learn graphviz
Run the Project
python task5.py
Implementation
Decision Tree
•	Trained a Decision Tree classifier.
•	Visualized the tree structure.
•	Controlled overfitting using max_depth.
Random Forest
•	Trained a Random Forest classifier.
•	Compared accuracy with the Decision Tree model.
•	Analyzed feature importance.
Model Evaluation
•	Accuracy Score
•	Classification Report
•	Confusion Matrix
•	5-Fold Cross Validation
Output Files
File	Description
decision_tree.png	Visualization of the trained Decision Tree
feature_importance.png	Feature importance plot from Random Forest
Key Learnings
•	Understanding Decision Tree algorithms
•	Preventing overfitting through depth control
•	Ensemble learning with Random Forests
•	Feature importance interpretation
•	Model evaluation using Cross-Validation
Conclusion
The Random Forest model generally provides better accuracy and generalization than a single Decision Tree by combining predictions from multiple trees. This project highlights the strengths of tree-based machine learning algorithms for classification tasks.


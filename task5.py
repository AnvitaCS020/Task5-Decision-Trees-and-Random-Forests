# ============================================================
# Task 5: Decision Trees and Random Forests
# Dataset: Heart Disease Classification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.tree import (
    DecisionTreeClassifier,
    export_graphviz,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("heart.csv")

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ============================================================
# 2. Separate Features and Target
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

# ============================================================
# 3. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. Decision Tree (Without Depth Control)
# ============================================================

dt_full = DecisionTreeClassifier(random_state=42)

dt_full.fit(X_train, y_train)

train_pred_full = dt_full.predict(X_train)
test_pred_full = dt_full.predict(X_test)

train_acc_full = accuracy_score(y_train, train_pred_full)
test_acc_full = accuracy_score(y_test, test_pred_full)

print("\n========== Decision Tree (Full Tree) ==========")
print("Training Accuracy :", round(train_acc_full, 4))
print("Testing Accuracy  :", round(test_acc_full, 4))

# ============================================================
# 5. Decision Tree (Controlled Depth)
# ============================================================

dt_pruned = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

dt_pruned.fit(X_train, y_train)

train_pred_pruned = dt_pruned.predict(X_train)
test_pred_pruned = dt_pruned.predict(X_test)

train_acc_pruned = accuracy_score(y_train, train_pred_pruned)
test_acc_pruned = accuracy_score(y_test, test_pred_pruned)

print("\n========== Decision Tree (Max Depth = 4) ==========")
print("Training Accuracy :", round(train_acc_pruned, 4))
print("Testing Accuracy  :", round(test_acc_pruned, 4))

# ============================================================
# 6. Random Forest
# ============================================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n========== Random Forest ==========")
print("Accuracy :", round(rf_accuracy, 4))

# ============================================================
# 7. Classification Report
# ============================================================

print("\nClassification Report (Random Forest)\n")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, rf_pred))

# ============================================================
# 8. Cross Validation
# ============================================================

dt_cv = cross_val_score(
    dt_pruned,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

rf_cv = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n========== Cross Validation ==========")

print(
    "Decision Tree CV Accuracy:",
    round(dt_cv.mean(), 4)
)

print(
    "Random Forest CV Accuracy:",
    round(rf_cv.mean(), 4)
)

# ============================================================
# 9. Feature Importance (Random Forest)
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== Feature Importance ==========")
print(importance)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ============================================================
# 10. Visualize Decision Tree
# ============================================================

plt.figure(figsize=(20, 10))

plot_tree(
    dt_pruned,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.savefig("decision_tree.png")
plt.show()

# ============================================================
# 11. Export Tree for Graphviz
# ============================================================

export_graphviz(
    dt_pruned,
    out_file="tree.dot",
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True
)

print("\nGraphviz file exported: tree.dot")

print("\nTask Completed Successfully!")
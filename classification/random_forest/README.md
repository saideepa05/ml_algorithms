# Random Forest — From Scratch (Concept + Math + Implementation)

This folder contains an implementation of a **Random Forest Classifier** built completely from scratch using **NumPy**, applied to the Heart Disease Prediction dataset.

The goal is to understand exactly how Random Forest works under the hood:
*   How trees are built.
*   How randomness helps (Bagging).
*   How predictions combine.
*   How we evaluate the model (Accuracy, Confusion Matrix, ROC–AUC).

**Random Forest is one of the most powerful algorithms for tabular classification.**

---

## 1. What is Random Forest?

A Random Forest is simply a **collection of many Decision Trees**, each trained on different random parts of the dataset. Their individual predictions are combined by **majority vote**.

Random Forest fixes the biggest weakness of a single Decision Tree: **Overfitting**.

### Why does it work so well?
1.  Trees are trained on different subsets of the data.
2.  Each tree looks at only some of the features.
3.  They are weak individually, but **strong together**.

This concept is called **Bagging** (Bootstrap Aggregating).

---

## 2. Key Concepts Behind Random Forest

### 2.1 Bootstrapping (Random Rows)
Each tree trains on a **random sample** of the dataset:
*   Sample with **replacement**.
*   Each tree gets a slightly different dataset.
*   This promotes diversity.

> **Example:** If the dataset has 300 rows, each tree trains on 300 random rows (some rows may appear multiple times, others not at all).

### 2.2 Random Feature Selection (Random Columns)
At **every split** in every tree, instead of checking *all* features, the tree only checks a **random subset**.

> **Example:** If the dataset has 10 features, a tree node might check only 3 random features to find the best split. This ensures trees do **not** look identical.

### 2.3 Building Each Decision Tree
Each tree uses the same logic as a standard Decision Tree:
1.  Calculate **Gini Impurity**.
2.  Calculate **Weighted Gini** and **Information Gain**.
3.  Find the best threshold.
4.  Recursively build child nodes.
5.  Stop at `max_depth` or small sample sizes.

#### Gini Impurity Formula
For binary classification ($p_0$ = class 0 fraction, $p_1$ = class 1 fraction):

$$ G = 1 - (p_0^2 + p_1^2) $$

---

### 2.4 Weighted Gini After Split
If a threshold divides the dataset into left ($L$) and right ($R$):

$$ G_{split} = p_L G_L + p_R G_R $$

Where:
*   $G_L, G_R$ = impurity of left & right nodes.
*   $p_L, p_R$ = fraction of samples in left & right.

---

### 2.5 Information Gain
Random Forest chooses the split that reduces impurity the most:

$$ \text{Gain} = G_{parent} - G_{split} $$

The best split has the **highest Gain**.

---

## 3. How Prediction Works (Majority Voting)

Suppose we have 10 trees and they predict for a single patient:

`[1, 0, 1, 1, 0, 1, 1, 0, 1, 1]`

*   Count of "1"s: 7
*   Count of "0"s: 3
*   **Final Prediction:** 1 (Heart Disease)

### Probability Estimate
The model can also output a probability score:

$$ P(y=1) = \frac{\text{number of trees predicting 1}}{\text{total trees}} $$

---

## 4. Why Random Forest Is So Powerful

| Feature | Single Decision Tree | Random Forest |
| :--- | :--- | :--- |
| **Variance** | High (Unstable) | Low (Stable) |
| **Overfitting** | Overfits easily | Resists overfitting |
| **Noise** | Memorizes noise | Noise cancels out |
| **Structure** | One complex model | Many simple models |
| **Reliability** | Variable | Highly Reliable |

**Conclusion:** Random Forest is one of the best "default" algorithms for tabular machine learning because it requires very little tuning to perform well.

# Decision Tree — From Scratch (Mathematical Explanation)

This repository contains an implementation of a Decision Tree Classifier from scratch using **NumPy**, applied to a Heart Disease Prediction dataset.

This README explains the intuition, mathematics, and learning process behind decision trees without using any external ML libraries.

## The Goal
**Understand exactly how a decision tree works** — how it splits data, how impurity is measured, and how the model grows recursively.

---

## 1. What a Decision Tree Does

A Decision Tree is a binary classification algorithm. Given patient features (age, cholesterol, blood pressure…), it predicts:

$$ \hat{y} \in \{0, 1\} $$

Where:
*   $y=1 \rightarrow$ heart disease
*   $y=0 \rightarrow$ no heart disease

A decision tree works by asking simple questions that split the data into purer groups.

---

## 2. Measuring Impurity (Gini Impurity)

To decide whether a split is good or bad, the tree computes the **Gini Impurity** of a node. Gini impurity measures how "mixed" the classes are.

For a node with class probabilities $p_1, p_2, ..., p_k$:

$$ G = 1 - \sum_{i=1}^{k} p_i^2 $$

For **binary classification** (0 and 1):

$$ G = 1 - (p_0^2 + p_1^2) $$

### Examples:
*   If all labels are the same $\rightarrow G = 0$ (Pure)
*   If labels are 50/50 $\rightarrow G = 0.5$ (Impure)
*   If labels are 70/30 $\rightarrow G = 0.42$

**Lower impurity = better.**

---

## 3. Splitting the Data

To split the dataset, the tree checks a specific feature $f$ and a threshold $t$.

*   **Left split:** $X_L = \{x : x_f \leq t\}$
*   **Right split:** $X_R = \{x : x_f > t\}$

Each possible threshold creates a different split. The objective is to find the split that reduces impurity the most.

---

## 4. Information Gain (Choosing the Best Split)

Before splitting, the impurity of the current (parent) node is $G_{parent}$.

After splitting into left and right nodes, the weighted average impurity is:

$$ G_{split} = p_L \cdot G_L + p_R \cdot G_R $$

Where:
*   $p_L = \frac{|X_L|}{|X|}$ (fraction of samples in left child)
*   $p_R = \frac{|X_R|}{|X|}$ (fraction of samples in right child)

The **Information Gain** is:

$$ \text{Gain} = G_{parent} - G_{split} $$

A **high gain** means the split made the data purer. The tree greedily chooses the feature + threshold that produces the **highest information gain**.

---

## 5. Building the Tree (Recursion)

A decision tree is built recursively.

**A node becomes a leaf if:**
1.  All labels are the same (impurity is 0).
2.  The maximum depth is reached.
3.  Too few samples are left to split.

The predicted label of a leaf is the majority class:

$$ \text{leaf\_value} = \text{argmax}(\text{count of each class}) $$

**If not a leaf:**
1.  Find the best feature and threshold.
2.  Split the data into left and right subsets.
3.  Recursively build the **left** subtree.
4.  Recursively build the **right** subtree.

---

## 6. Making Predictions

To classify a new point $x$, the tree traverses from the root down to a leaf by checking thresholds.

**Logic flow at each node:**
```python
if x[feature] <= threshold:
    go left
else:
    go right

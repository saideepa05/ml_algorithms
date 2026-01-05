# Regression Trees — From Scratch (CART)

This folder contains an implementation of a **Regression Tree** from scratch.

This README explains the intuition, the MSE loss function, and how the tree recursively splits data to predict continuous values.

## 1. What is a Regression Tree?

A Regression Tree predicts a continuous value by:
1.  **Recursively splitting** the feature space into rectangular regions.
2.  **Creating regions** where predictions are constant.
3.  **Using the mean target value** of the training samples in each leaf as the prediction.

**Unlike Linear Models:**
*   No linearity assumption.
*   No need for feature scaling.
*   Handles non-linear relationships naturally.

---

## 2. Model Intuition

A regression tree learns a hierarchy of **if-else rules**. Ideally, it partitions the data so that samples with similar target values end up in the same "leaf".

**Example Logic:**
```python
if area <= 1200:
    if bedrooms <= 2:
        predict 180,000
    else:
        predict 210,000
else:
    predict 350,000

---

## 3. Loss Function (MSE)

Regression trees minimize **Mean Squared Error (MSE)** (or Variance) to decide where to split.

**MSE at a specific node:**
For a node with target values $y_1, \dots, y_n$:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2 $$

Where:
*   $\bar{y} = \frac{1}{n} \sum y_i$ is the mean value of the node.

---

## 4. Best Split Criterion (The Math)

To find the best split, the algorithm tries every possible **feature $j$** and **threshold $t$**.

It splits the data into two groups:
*   **Left Node ($L$):** $X_j \leq t$
*   **Right Node ($R$):** $X_j > t$

It then calculates the **Weighted MSE Cost** of the split:

$$ \text{Cost}(j, t) = \frac{n_L}{n} \text{MSE}_L + \frac{n_R}{n} \text{MSE}_R $$

Where:
*   $n_L, n_R$ are the number of samples in left/right nodes.
*   $n$ is the total number of samples in the current node.

**The Goal:** Choose the $(j, t)$ pair that minimizes this Cost.

---

## 5. Stopping Criteria

Tree growth must stop to prevent overfitting (learning the noise). Common criteria include:

1.  **Maximum Depth:** The tree reaches a certain height.
2.  **Minimum Samples:** A node has too few samples to split further (e.g., < 2).
3.  **Pure Node:** All target values in the node are identical (MSE = 0).
4.  **No Improvement:** The best split does not significantly reduce the cost.

---

## 6. Prediction Rule

To predict for a new sample $x$:

1.  Start at the **Root**.
2.  Check the split condition (e.g., $x_{age} \leq 30$).
3.  Move **Left** or **Right** based on the result.
4.  Repeat until a **Leaf** is reached.
5.  **Output:** Return the stored mean value of that leaf.

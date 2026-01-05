# Lasso Regression (L1 Regularization) — From Scratch

This folder contains an implementation of **Lasso Regression** from scratch.

This README explains the intuition, the L1 loss function, why standard gradient descent struggles with Lasso, and how **Coordinate Descent** with **Soft-Thresholding** is used to solve it.

## 1. What is Lasso Regression?

**Lasso** (Least Absolute Shrinkage and Selection Operator) is a linear model that adds an **L1 penalty** on the weights. This encourages **sparsity**, meaning it forces less important feature weights to become exactly **zero**.

**It is used when:**
*   Feature selection is desired (automatic pruning).
*   Many features might be irrelevant or noisy.
*   Model interpretability matters.

**Difference from Ridge Regression:**
*   Ridge shrinks weights *near* zero but never *exactly* to zero.
*   **Lasso can force weights to exactly 0**, effectively removing the feature from the model.

---

## 2. Model Formulation

The prediction function is the same as ordinary Linear Regression:

$$ \hat{y} = Xw + b $$

---

## 3. Loss Function (Lasso / L1)

The objective function combines the Mean Squared Error (MSE) with the L1 penalty:

$$ J_{lasso}(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^{d} |w_j| $$

Where:
*   **L1 penalty:** $\sum |w_j|$ (Sum of absolute values of weights).
*   **$\lambda$ (Lambda):** Controls the strength of regularization.
    *   High $\lambda \rightarrow$ More weights become zero (sparser model).
    *   $\lambda = 0 \rightarrow$ Standard Linear Regression.

---

## 4. Why Gradient Descent is Tricky for Lasso

The absolute value function $|w|$ is **not differentiable at 0** (it has a sharp corner). This makes standard Gradient Descent unstable or impossible for finding exact zeros.

Instead, the standard industry approach to optimize Lasso is **Coordinate Descent** using **Soft-Thresholding**.

---

## 5. Soft-Thresholding Operator

This is the mathematical operation that determines the new weight. It shrinks the value towards zero and snaps it to zero if it is small enough.

For a specific value $z$ and threshold $\gamma$, the operator $S(z, \gamma)$ is:

$$ S(z, \gamma) = \begin{cases} z - \gamma & \text{if } z > \gamma \\ 0 & \text{if } |z| \leq \gamma \\ z + \gamma & \text{if } z < -\gamma \end{cases} $$

**Intuition:**
*   If the weight is large and positive, subtract a little bit.
*   If the weight is large and negative, add a little bit.
*   If the weight is small (inside $[-\gamma, \gamma]$), **kill it (set to 0)**.

---

## 6. Coordinate Descent Algorithm

Instead of updating all weights at once (like Gradient Descent), **Coordinate Descent** updates one weight $w_j$ at a time while holding all others fixed.

**For each feature $j$:**
1.  Remove the contribution of feature $j$ from the predictions.
2.  Compute the **partial residual** (how much error is left for this feature to explain).
3.  Apply the **Soft-Thresholding** operator to find the new $w_j$.
4.  **Bias ($b$)** is updated separately using standard mean error (bias is not regularized).

This process repeats until the weights converge.

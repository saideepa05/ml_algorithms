# Support Vector Machine (SVM) — From Scratch (Mathematical Explanation)

This folder contains an implementation of a **Support Vector Machine (SVM)** classifier from scratch using **NumPy**.

This README explains the intuition, mathematics, learning rule, hinge loss, and gradient updates behind SVM — step by step, in clean mathematical notation.

## The Goal
**Understand exactly how SVM works** → How it finds a separating hyperplane, what the margin means, how hinge loss punishes mistakes, and how the gradient update works.

---

## 1. What SVM Does

SVM is a binary classifier that tries to separate two classes using a:
*   **Line** (2D)
*   **Plane** (3D)
*   **Hyperplane** (Higher dimensions)

Given input features $x$ and label $y \in \{-1, +1\}$, SVM predicts:

$$ \hat{y} = \text{sign}(w^T x + b) $$

Where:
*   $w$ = weight vector (orientation)
*   $b$ = bias (position)
*   $w^T x + b = 0$ is the **decision boundary**.

SVM wants to find the **best boundary** — the one that maximizes the empty space (**margin**) between the two classes.

---

## 2. The Decision Boundary

The boundary is defined by:
$$ w^T x + b = 0 $$

The regions are defined as:
*   If $w^T x + b > 0 \Rightarrow$ Class **+1**
*   If $w^T x + b < 0 \Rightarrow$ Class **-1**

So the final prediction is:

$$ \hat{y} = \text{sign}(w^T x + b) $$

---

## 3. The Margin — The Heart of SVM

The size of the margin is:

$$ \text{margin} = \frac{2}{\|w\|} $$

SVM wants to **maximize the margin**, which is mathematically equivalent to **minimizing the weights**:

$$ \min \frac{1}{2} \|w\|^2 $$

This ensures the simplest, cleanest possible boundary (Regularization).

---

## 4. 🧩 Perfect Separation (Ideal Case)

If data is perfectly separable, we want all points to satisfy:

$$ y_i (w^T x_i + b) \geq 1 $$

Points that satisfy the equality exactly:
$$ y_i (w^T x_i + b) = 1 $$
are called **Support Vectors**. These are the specific data points that "hold up" or define the boundary.

---

## 5. 💥 Real Life: Soft-Margin SVM

Real data is messy and rarely perfectly separable. SVM handles this by introducing **Hinge Loss**:

$$ \max(0, \ 1 - y_i(w^T x_i + b)) $$

**Interpretation:**

| Condition | Meaning | Loss |
| :--- | :--- | :--- |
| $y_i(w^T x_i + b) \geq 1$ | Correct & Safe (Outside margin) | **0** |
| $0 < y_i(w^T x_i + b) < 1$ | Correct but too close (Inside margin) | **Small Penalty** |
| $y_i(w^T x_i + b) < 0$ | Wrong side (Misclassified) | **Big Penalty** |

---

## 6. 🎯 Full SVM Loss Function

The total cost function combines "keeping weights small" (Regularization) and "reducing mistakes" (Hinge Loss):

$$ L = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max(0, \ 1 - y_i(w^T x_i + b)) $$

*   $\frac{1}{2} \|w\|^2$ $\rightarrow$ Maximizes margin.
*   $C$ $\rightarrow$ How strictly we penalize mistakes.
*   **Hinge Loss** $\rightarrow$ Punishes misclassified or unsafe points.

---

## 7. 🔮 Final Prediction Rule

Once trained, for any new $x$:

$$ \hat{y} = \text{sign}(w^T x + b) $$

---

## 8. 🔥 GRADIENT UPDATE FOR SVM

This is the logic used inside the `fit` method. We use **Gradient Descent** to minimize the loss.

For every example $(x_i, y_i)$, we check the condition:

$$ y_i (w^T x_i + b) \geq 1 $$

There are two cases:

### ✅ Case 1: Margin $\geq$ 1 (Correct & Safe)
The point is correctly classified and far enough away. No hinge loss applies.
The gradient comes **only** from the regularization term $\|w\|^2$.

$$ dw = 2\lambda w $$
$$ db = 0 $$

**In Code:**
```python
if condition:
    dw = 2 * self.lambda_param * self.w
    db = 0

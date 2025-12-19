# XGBoost — From Scratch (Mathematical Explanation)

This folder contains an implementation of an **XGBoost-style classifier** from scratch using **NumPy**, applied to a binary classification problem (Heart Disease Prediction).

This implementation is not a wrapper around existing libraries. It explicitly implements the core ideas of XGBoost:
*   **Second-order optimization** (Newton boosting).
*   **Regularized tree learning**.
*   **Gain-based split selection**.

## The Goal
**Understand how XGBoost really works:**
*   How it improves upon Gradient Boosting.
*   Why it uses gradients and Hessians.
*   How leaf values are computed optimally.
*   How regularization controls model complexity.

---

## 1. What Is XGBoost?

XGBoost (**Extreme Gradient Boosting**) is an advanced boosting algorithm that builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous model.

Unlike classic Gradient Boosting, XGBoost:
1.  Uses **second-order derivatives** (Hessian).
2.  Applies **explicit regularization** ($\lambda, \gamma$).
3.  Optimizes splits using a specialized **Gain formula**.

---

## 2. What Does the Model Predict?

XGBoost does not directly predict labels. It learns a function $F(x)$ that predicts the **Log-Odds (Logit)**:

$$ F(x) \approx \log \left( \frac{P(y=1|x)}{1 - P(y=1|x)} \right) $$

To get the probability, we apply the **Sigmoid Function**:

$$ P(y=1|x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}} $$

**Final Classification:**

$$ \hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq 0.5 \\ 0 & \text{otherwise} \end{cases} $$

---

## 3. Loss Function (Binary Log-Loss)

XGBoost minimizes the **Binary Cross-Entropy Loss**:

$$ L(y, p) = -[y \log(p) + (1-y) \log(1-p)] $$

Where:
*   $y \in \{0, 1\}$ is the true label.
*   $p = P(y=1|x)$ is the predicted probability.

---

## 4. Second-Order Taylor Expansion (Key Idea)

Instead of minimizing the loss directly (like Gradient Descent), XGBoost uses a **Second-Order Taylor Approximation** around the current prediction $F$:

$$ L(F+f) \approx L(F) + g f + \frac{1}{2} h f^2 $$

Where:
*   $g = \frac{\partial L}{\partial F}$ (**Gradient** - 1st Derivative)
*   $h = \frac{\partial^2 L}{\partial F^2}$ (**Hessian** - 2nd Derivative)

For **Binary Log-Loss**, these are:

$$ g_i = p_i - y_i $$
$$ h_i = p_i (1 - p_i) $$

---

## 5. Why Gradients and Hessians Matter

*   **Gradient ($g$):** Tells us the direction of the error (how wrong we are).
*   **Hessian ($h$):** Tells us the curvature of the loss (how confident we are).

Using both allows XGBoost to take **smarter update steps** (Newton Boosting), leading to faster convergence than standard Gradient Boosting.

---

## 6. Tree Structure and Leaf Weights

Each tree outputs a constant value per leaf. The **Optimal Leaf Weight** is derived by minimizing the Taylor expansion:

$$ w^* = -\frac{\sum g_i}{\sum h_i + \lambda} $$

Where:
*   $\lambda$ is the **L2 Regularization** parameter.
*   It prevents the leaf weights from becoming too large (overfitting).

---

## 7. Split Selection (Gain Formula)

To decide the best feature and threshold to split on, XGBoost calculates the **Gain** (Structure Score Improvement):

$$ \text{Gain} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} - \gamma $$

Where:
*   $G = \sum g_i$ (Sum of Gradients)
*   $H = \sum h_i$ (Sum of Hessians)
*   $\gamma$ (**Gamma**) penalizes the creation of new leaves.

The split with the **Maximum Gain** is chosen.

---

## 8. Regularization Parameters

| Parameter | Symbol | Effect |
| :--- | :--- | :--- |
| **Lambda** | $\lambda$ | **L2 Regularization** on leaf weights. Controls magnitude. |
| **Gamma** | $\gamma$ | **Minimum Loss Reduction**. Penalizes making new splits. |

These two parameters are the main reason why XGBoost generalizes better than standard boosting.

---

## 9. Training Algorithm (Step-by-Step)

**Step 1: Initialize**
Start with a constant prediction (log-odds of the mean):
$$ F_0 = \log \left( \frac{\bar{y}}{1 - \bar{y}} \right) $$

**Step 2: Boosting Loop (for $m = 1 \dots M$)**

1.  **Compute Probabilities:** $p_i = \sigma(F_{m-1}(x_i))$
2.  **Compute Gradients:** $g_i = p_i - y_i$
3.  **Compute Hessians:** $h_i = p_i (1 - p_i)$
4.  **Fit a Tree:** Find the split that maximizes **Gain**.
5.  **Compute Leaf Weights:** $w^* = -\frac{\sum g}{\sum h + \lambda}$
6.  **Update Model:** $F_m(x) = F_{m-1}(x) + \eta f_m(x)$

---

## 10. Prediction Phase

After training, the final model is the sum of all trees:

$$ F(x) = F_0 + \sum_{m=1}^{M} \eta f_m(x) $$

We convert this to a probability for the final result:

$$ P(y=1|x) = \sigma(F(x)) $$

# SVR (Support Vector Regression) — Math (From Scratch)

This folder contains an implementation of **Support Vector Regression (SVR)** from scratch.

This README explains the goal of SVR, the **$\epsilon$-insensitive loss function**, the Primal optimization problem, and the subgradient descent updates used for training.

## 1. Goal

Given a dataset $\{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, SVR learns a linear function:

$$ f(x) = w^T x + b $$

The goal is to find a function that is "flat" (small $\|w\|$) while ensuring that predictions stay within an **$\epsilon$-tube** around the target values.

---

## 2. $\epsilon$-Insensitive Loss (Core Idea)

SVR ignores small errors. If the prediction is close enough to the actual value (within a margin $\epsilon$), the loss is zero.

$$ L_\epsilon(y, f(x)) = \max(0, \ |y - f(x)| - \epsilon) $$

**Interpretation:**
*   If error $\le \epsilon \rightarrow$ Loss = 0 (Safe zone).
*   If error $> \epsilon \rightarrow$ Loss increases linearly.

This makes SVR robust to noise and focuses only on significant errors.

---

## 3. Primal Optimization Problem (Soft-Margin SVR)

To allow for some errors (soft margin), SVR introduces slack variables $\xi_i, \xi_i^*$ and minimizes:

$$ \min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*) $$

**Subject to:**
1.  $y_i - (w^T x_i + b) \le \epsilon + \xi_i$
2.  $(w^T x_i + b) - y_i \le \epsilon + \xi_i^*$
3.  $\xi_i \ge 0, \ \xi_i^* \ge 0$

**Key Parameters:**
*   **$\frac{1}{2} \|w\|^2$:** Regularization term (encourages a flat function).
*   **$C$:** Penalty parameter. Bigger $C$ = Stricter fit (less tolerance for errors outside the tube).
*   **$\epsilon$:** Tube width. Bigger $\epsilon$ = More tolerance (wider safety zone).

---

## 4. Equivalent Unconstrained Objective

We can rewrite the constrained problem as a single unconstrained cost function (regularization + loss):

$$ J(w, b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max(0, \ |y_i - (w^T x_i + b)| - \epsilon) $$

This is the form we optimize using **Subgradient Descent**.

---

## 5. Subgradients (Implementation Logic)

Let the residual be $r_i = y_i - (w^T x_i + b)$.
The loss is only active if the absolute residual $|r_i|$ is greater than $\epsilon$.

There are three cases for the gradient update:

### Case 1: Within the Tube ($|r_i| \le \epsilon$)
The point is "correct enough". No update needed from the loss term.

$$ \frac{\partial \text{Loss}}{\partial w} = 0, \quad \frac{\partial \text{Loss}}{\partial b} = 0 $$

### Case 2: Under-predicting ($r_i > \epsilon$)
The prediction is too low ($y_i$ is much higher).

$$ \frac{\partial \text{Loss}}{\partial w} = -C x_i $$

$$ \frac{\partial \text{Loss}}{\partial b} = -C $$

### Case 3: Over-predicting ($r_i < -\epsilon$)
The prediction is too high ($y_i$ is much lower).

$$ \frac{\partial \text{Loss}}{\partial w} = +C x_i $$

$$ \frac{\partial \text{Loss}}{\partial b} = +C $$

**Total Update Rule:**
We combine the regularization gradient ($\frac{\partial}{\partial w} \frac{1}{2}\|w\|^2 = w$) with the loss gradient derived above.

$$ w \leftarrow w - \alpha (\text{Regularization Grad} + \text{Loss Grad}) $$

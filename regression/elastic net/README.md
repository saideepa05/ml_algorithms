# Elastic Net Regression — From Scratch

This folder contains an implementation of **Elastic Net Regression** from scratch.

This README explains the intuition, the combined L1/L2 loss function, and the **Coordinate Descent** algorithm used to optimize it.

## 1. What is Elastic Net?

Elastic Net is a regularized linear regression model that combines the best of both worlds:
*   **L1 penalty (Lasso):** Encourages sparsity (feature selection).
*   **L2 penalty (Ridge):** Stabilizes coefficients and handles multicollinearity (highly correlated features).

**It is useful when:**
*   Many features are correlated (Lasso tends to pick just one randomly; Elastic Net picks both).
*   You want sparsity (like Lasso) but better stability (like Ridge).

---

## 2. Model Formulation

The prediction model is the standard linear equation:

$$ \hat{y} = Xw + b $$

Where:
*   $X \in \mathbb{R}^{n \times d}$
*   $w \in \mathbb{R}^d$
*   $b \in \mathbb{R}$

---

## 3. Elastic Net Loss Function

Elastic Net adds both **L1** and **L2** penalties to the Mean Squared Error (MSE):

$$ J_{EN}(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 + \lambda \left( \alpha \sum_{j=1}^{d} |w_j| + (1 - \alpha) \sum_{j=1}^{d} w_j^2 \right) $$

**Parameters:**
*   **$\lambda \geq 0$:** Overall regularization strength.
*   **$\alpha \in [0, 1]$:** The mixing ratio between L1 and L2.

**Special Cases:**
*   $\alpha = 1 \rightarrow$ **Lasso**
*   $\alpha = 0 \rightarrow$ **Ridge**
*   $0 < \alpha < 1 \rightarrow$ **Elastic Net**

⚠️ **Note:** The bias term $b$ is not regularized.

---

## 4. Optimization Method (Coordinate Descent)

Because of the L1 term ($|w_j|$), the derivative is undefined at 0. Therefore, we cannot use standard Gradient Descent. Instead, we use **Coordinate Descent** with **Soft-Thresholding**.

**Soft-Thresholding Operator:**
This operator shrinks values towards zero and sets them exactly to zero if they are within the threshold $\gamma$.

$$ S(z, \gamma) = \begin{cases} z - \gamma & \text{if } z > \gamma \\ 
0 & \text{if } |z| \leq \gamma \\
z + \gamma & \text{if } z < -\gamma \end{cases} $$

---

## 5. Coordinate Update (The Math)

In Coordinate Descent, we update one weight $w_j$ at a time while keeping others fixed.

Let $x_j$ be the column $j$ of feature matrix $X$.
First, calculate the **partial residual** (the error remaining *without* the contribution of feature $j$):

$$ r = y - \left( b + \sum_{k \neq j} x_k w_k \right) $$

Next, calculate the correlation between feature $j$ and the residual:

$$ z = x_j^T r $$

**The Elastic Net Update Rule:**

$$ w_j \leftarrow \frac{S(z, \lambda \alpha n)}{\|x_j\|^2 + \lambda(1 - \alpha)n} $$

**Bias Update:**
The bias is updated using the mean residual:

$$ b \leftarrow \frac{1}{n} \sum_{i=1}^{n} (y_i - (Xw)_i) $$

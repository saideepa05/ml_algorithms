# Linear Regression — From Scratch (Mathematical Explanation)

This folder contains an implementation of **Linear Regression** from scratch using **NumPy**, applied to a regression dataset.

This README explains the intuition, the mathematical formulation, the loss function (MSE), and the gradient descent updates used to train the model.

## The Goal
**Understand exactly how Linear Regression works** → How it predicts continuous values, how it measures error, and how it learns the best-fit line (or hyperplane) using gradients.

---

## 1. What Linear Regression Does

Linear regression predicts a **continuous number** (not a class label).
Given input features $x = (x_1, x_2, \dots, x_d)$, we predict:

$$ \hat{y} = w^T x + b $$

Where:
*   $w \in \mathbb{R}^d$ = Weights (Slopes)
*   $b \in \mathbb{R}$ = Bias (Intercept)

For the full dataset matrix $X \in \mathbb{R}^{n \times d}$:

$$ \hat{y} = Xw + b $$

---

## 2. Loss Function (Mean Squared Error)

We need a way to measure how wrong the predictions are.
For a single data point, the error is:

$$ (\hat{y}_i - y_i)^2 $$

For all $n$ samples, we calculate the **Mean Squared Error (MSE)** cost function:

$$ J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 $$

**Goal:** Find $w$ and $b$ that minimize $J(w, b)$.

---

## 3. Gradient Descent Learning

We update parameters iteratively by moving in the direction **opposite** to the gradient (slope) of the loss function.

**Update Rules:**

$$ w \leftarrow w - \alpha \frac{\partial J}{\partial w} $$

$$ b \leftarrow b - \alpha \frac{\partial J}{\partial b} $$

Where $\alpha$ is the **Learning Rate**.

### Deriving the Gradients
First, define the error vector:
$$ e = \hat{y} - y = (Xw + b) - y $$

The gradients for MSE are derived as:

**Gradient w.r.t Weights ($w$):**
$$ \frac{\partial J}{\partial w} = \frac{2}{n} X^T (\hat{y} - y) $$

**Gradient w.r.t Bias ($b$):**
$$ \frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) $$

### Implementation in Code
This math translates directly into NumPy code:
```python
dw = (2/n) * np.dot(X.T, (y_pred - y))
db = (2/n) * np.sum(y_pred - y)

self.w -= self.lr * dw
self.b -= self.lr * db


---

## 4. Evaluation Metrics (Regression)

Unlike classification, we don’t use Accuracy or F1-Score. Instead, we use metrics that measure the **distance** between predictions and actual values.

### Mean Squared Error (MSE)
Average squared difference between predicted and actual values. It heavily penalizes large errors (outliers).

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 $$

### Root Mean Squared Error (RMSE)
The square root of MSE. This is useful because it is in the **same units** as the target variable (e.g., dollars, meters).

$$ \text{RMSE} = \sqrt{\text{MSE}} $$

### Mean Absolute Error (MAE)
The average of absolute differences. It gives a linear score and is **less sensitive to outliers** than MSE.

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i| $$

### R² Score (Coefficient of Determination)
Represents the proportion of variance in the dependent variable that is predictable from the independent variables.

*   $R^2 = 1$: Perfect prediction.
*   $R^2 = 0$: The model is no better than simply predicting the mean of the target for every point.

$$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$

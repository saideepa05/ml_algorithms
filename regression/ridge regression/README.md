# Ridge Regression (L2 Regularization) — From Scratch

This folder contains an implementation of **Ridge Regression** from scratch using **NumPy**.

This README explains the intuition, the L2 loss function, the effect of $\lambda$, and the gradient descent updates used to train the model.

## 1. What is Ridge Regression?

Ridge Regression is a **regularized** version of Linear Regression that adds a penalty on the magnitude of the weights.

**It is used when:**
*   Features are highly correlated (**Multicollinearity**).
*   Standard Linear Regression overfits the training data.
*   We want smaller, more stable coefficients.

**Key Characteristic:**
Ridge **shrinks** weights towards zero, but unlike Lasso, it does **not** set them exactly to zero.

---

## 2. Model Formulation

The prediction function is the same as ordinary Linear Regression:

$$ \hat{y} = Xw + b $$

Where:
*   $X \in \mathbb{R}^{n \times d}$ $\rightarrow$ Feature matrix.
*   $w \in \mathbb{R}^d$ $\rightarrow$ Weights vector.
*   $b \in \mathbb{R}$ $\rightarrow$ Bias term.

---

## 3. Loss Function (Ridge / L2)

Ridge adds an **L2 penalty** to the Mean Squared Error (MSE):

$$ J_{ridge}(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^{d} w_j^2 $$

Where:
*   **First term:** Measures how well the model fits the data (MSE).
*   **Second term:** Regularization penalty (Sum of squared weights).
*   **$\lambda \geq 0$:** Controls the strength of regularization.

**Effect of $\lambda$:**
*   $\lambda = 0$: Becomes ordinary Linear Regression.
*   Small $\lambda$: Slight shrinkage of weights.
*   Large $\lambda$: Strong shrinkage (weights become very small).

**Note:** The bias term $b$ is usually **not** regularized.

---

## 4. Gradient Derivation

To optimize the model, we use Gradient Descent. First, define the error vector:
$$ e = \hat{y} - y $$

**Gradient w.r.t Weights ($w$):**
The derivative includes the penalty term $2\lambda w$:

$$ \frac{\partial J}{\partial w} = \frac{2}{n} X^T e + 2\lambda w $$

**Gradient w.r.t Bias ($b$):**
The bias is not penalized, so its derivative is standard:

$$ \frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i $$

---

## 5. Optimization Method

Ridge Regression is trained using **Batch Gradient Descent**. The update rules are:

**Update Weights:**

$$
w \leftarrow w - \alpha \left( \frac{2}{n} X^T e + 2\lambda w \right) 
$$

**Update Bias:**

$$ 
b \leftarrow b - \alpha \left( \frac{2}{n} \sum e_i \right) 
$$

Where $\alpha$ is the learning rate.

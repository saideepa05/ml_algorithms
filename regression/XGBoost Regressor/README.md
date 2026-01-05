# Gradient Boosting Regressor — From Scratch

This folder contains an implementation of **Gradient Boosting Regression** from scratch.

This README explains how boosting sequentially builds trees to correct errors, the MSE loss function, and the mathematical algorithm behind gradient-based learning.

## 1. What Problem Gradient Boosting Solves

**Random Forest** reduces **variance** by averaging many independent trees (Bagging).
But sometimes the problem is **bias** (underfitting), not variance.

**Gradient Boosting Regression solves this by:**
*   Building trees **sequentially**.
*   Each new tree **corrects the mistakes** of the previous ones.
*   Instead of many independent learners, it creates one strong model by adding weak learners step by step.

---

## 2. Core Idea (Plain Language)

Gradient Boosting works like this:
1.  Start with a simple model (e.g., predict the average).
2.  **Measure its errors** (residuals).
3.  Train a **new model** to fix those errors.
4.  **Add** the new model to the existing one.
5.  Repeat.

Each new tree focuses on what the model is still getting wrong.

---

## 3. Model Formulation

The model is built as an **additive function**:

$$ f_0(x) \rightarrow f_1(x) \rightarrow \dots \rightarrow f_M(x) $$

**Final Model:**

$$ f_M(x) = \sum_{m=0}^{M} \eta h_m(x) $$

Where:
*   $h_m(x)$ = Regression Tree (Weak Learner).
*   $\eta$ = Learning Rate (Step size).
*   $M$ = Number of trees.

---

## 4. Loss Function (Regression)

For regression, Gradient Boosting typically minimizes **Mean Squared Error (MSE)**. We often use a factor of $1/2$ to simplify the derivative:

$$ L(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2 $$

---

## 5. Gradient Boosting Algorithm (Step-by-Step)

Given training data $\{(x_i, y_i)\}_{i=1}^n$:

### Step 1: Initialize the Model
Start with a constant prediction that minimizes MSE (the mean):

$$ f_0(x) = \text{argmin}_c \sum (y_i - c)^2 = \bar{y} $$

So initially: $\hat{y}_i^{(0)} = \bar{y}$

### Step 2: Compute Residuals (Negative Gradients)
Calculate the derivative of the loss function with respect to the prediction:

$$ r_i^{(m)} = -\frac{\partial L}{\partial \hat{y}_i} = y_i - \hat{y}_i^{(m-1)} $$

These **residuals** represent exactly what the model is currently missing.

### Step 3: Fit a Regression Tree to Residuals
Train a shallow regression tree $h_m(x)$ to predict these residuals:

$$ h_m(x) \approx r^{(m)} $$

These trees are usually **weak learners** (small depth).

### Step 4: Update the Model
Add the new tree to the existing model, scaled by the learning rate:

$$ f_m(x) = f_{m-1}(x) + \eta h_m(x) $$

Where $\eta \in (0, 1]$ is the **Learning Rate**.

### Step 5: Repeat
Repeat Steps 2–4 for $M$ iterations. Each step improves the model incrementally.

---

## 6. Prediction Rule

For a new input $x$, the final prediction is the sum of the initial mean plus all the weighted tree corrections:

$$ \hat{y}(x) = f_0(x) + \sum_{m=1}^{M} \eta h_m(x) $$

---

## 7. Why Gradient Boosting Works

Gradient Boosting succeeds because it:
*   **Directly optimizes** the loss function using gradient descent in function space.
*   **Focuses** on hard-to-predict samples (large residuals).
*   **Reduces Bias** significantly.
*   Builds complex non-linear functions from simple trees.

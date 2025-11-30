# Logistic Regression (From Scratch) – Intuition + Math

This project implements **Logistic Regression from scratch** using NumPy (no `sklearn` model) and applies it to a **Heart Disease Prediction** dataset.

Logistic Regression is a **binary classification** algorithm.  
It answers questions like:

> “Given patient features (age, cholesterol, blood pressure, etc.), what is the probability that this patient has heart disease (1) versus not (0)?”

---

## 1. High-Level Idea

The model works in three main steps:

1. **Combine features linearly** → produce a raw score \( z \)
2. **Squash the score into a probability** between 0 and 1 using the **sigmoid** function
3. **Decide class 0 or 1** by applying a threshold (usually 0.5)

---

## 2. Step 1 – Linear Score

Let:

- \( x \in \mathbb{R}^d \) be the input feature vector  
  (e.g., age, sex, chest pain type, blood pressure, cholesterol, etc.)
- \( w \in \mathbb{R}^d \) be the weight vector (one weight per feature)
- \( b \in \mathbb{R} \) be the bias term (intercept)

We first compute a **linear combination** of the features:

\[
z = w^\top x + b
\]

- \(w^\top x\) means: \(w_1 x_1 + w_2 x_2 + \dots + w_d x_d\)
- This raw score \(z\) can be any real number (negative, positive, large, small).

**Code equivalent:**

```python
z = np.dot(X, self.weights) + self.bias

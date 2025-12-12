# Logistic Regression — From Scratch (Mathematical Explanation)

This contains an implementation of **Logistic Regression from scratch** using NumPy, applied to a Heart Disease Prediction dataset.  
This README explains the **intuition**, **mathematics**, and **learning process** behind logistic regression without using any external ML libraries.

The goal is simple:

Understand exactly how logistic regression works — how it makes predictions, how we measure error, and how the model learns through gradient descent.

# 1. What Logistic Regression Does
Logistic Regression is a **binary classification** algorithm.

Given patient features (age, cholesterol, blood pressure…), it predicts

$$
P(y = 1 \mid x)
$$

where:
- $y = 1$ → heart disease  
- $y = 0$ → no heart disease  

The model outputs a **probability** between 0 and 1 and then converts it into a class (0 or 1).

# 2. The Model: Three Steps
Logistic Regression works in **three steps**:

### 1. Compute a linear score  
$$
z = w^\top x + b
$$

### 2. Squash it into a probability  
$$
\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

### 3. Convert probability → class  
$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

# 3. Step 1 — Linear Score
Let:

- $x \in \mathbb{R}^d$: feature vector  
- $w \in \mathbb{R}^d$: weights  
- $b \in \mathbb{R}$: bias  
The model computes:

$$
z = w^\top x + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b
$$

This number $z$ can be any real value — positive, negative, small, or large.

# 4. Step 2 — Sigmoid: Score → Probability
The **sigmoid function** converts the raw score into a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Properties:
- If $z$ is large and positive → $\sigma(z) \approx 1$  
- If $z$ is large and negative → $\sigma(z) \approx 0$  
- If $z = 0$ → $\sigma(0) = 0.5$  
Thus,

$$
\hat{p}(y=1 \mid x) = \sigma(w^\top x + b)
$$

This is the core of logistic regression.

# 5. Step 3 — Probability → Class
To make a final decision:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \ge 0.5 \\
0 & \text{if } \hat{p} < 0.5
\end{cases}
$$

# 6. Log-Odds Interpretation (Why “Logistic”?)
Let:
- $p = P(y=1 \mid x)$  
- $1 - p = P(y=0 \mid x)$  

**Odds:**

$$
\text{odds} = \frac{p}{1-p}
$$

**Log-odds (logit):**

$$
\log\left( \frac{p}{1-p} \right)
$$

Logistic regression assumes:

$$
\log\left(\frac{p}{1-p}\right) = w^\top x + b
$$

Solving for $p$:

$$
p = \frac{1}{1 + e^{-(w^\top x + b)}}
$$

This gives back the **sigmoid**, so logistic regression models **linear log-odds**.

# 7. Binary Cross-Entropy Loss
To train the model, we need a way to measure **how wrong** predictions are.
For one example:
- true label: $y \in \{0, 1\}$  
- predicted probability: $p$

The binary cross-entropy loss is:

$$
\ell(y, p)
= -\left[ y \log(p) + (1 - y)\log(1 - p) \right]
$$

Intuition:
- If $y = 1$ and $p$ is close to 1 → loss is small  
- If $y = 1$ and $p$ is close to 0 → loss explodes  
- If $y = 0$ and $p$ is close to 0 → small  
- If $y = 0$ and $p$ is close to 1 → huge  


For all $n$ samples:
J(w, b) = (1/n) * Σ from i=1 to n [
    - y(i) * log(p(i))
    - (1 - y(i)) * log(1 - p(i))
]

This is the **cost function** we minimize.

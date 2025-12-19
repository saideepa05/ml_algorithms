# Gradient Boosting Classifier — From Scratch (Mathematical Explanation)

This folder contains an implementation of a **Gradient Boosting Classifier** from scratch using **NumPy**, applied to a binary classification problem (Heart Disease Prediction).

**No external machine learning libraries are used for the model itself.**

---

## 1. What Is Gradient Boosting?

Gradient Boosting is an ensemble learning method that builds a strong model by adding many weak models **sequentially**.

*   Each new model is trained to **correct the mistakes** made by the current ensemble.
*   Instead of learning everything at once, Gradient Boosting learns small corrections step by step.

---

## 2. What Does the Model Predict?

For binary classification, Gradient Boosting does **not** directly predict class labels. Instead, it learns a function $F(x)$ that outputs the **Log-Odds (Logit)**:

$$ F(x) \approx \log \left( \frac{P(y=1|x)}{P(y=0|x)} \right) $$

To get the actual probability, we pass this output through the **Sigmoid Function**:

$$ P(y=1|x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}} $$

**Final Prediction:**

$$ \hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq 0.5 \\ 0 & \text{otherwise} \end{cases} $$

---

## 3. Loss Function (Binary Log-Loss)

Gradient Boosting minimizes the **Binary Cross-Entropy Loss**:

$$ L(y, p) = -[y \log(p) + (1-y) \log(1-p)] $$

Where:
*   $y \in \{0, 1\}$ is the true label.
*   $p = P(y=1|x)$ is the predicted probability.

---

## 4. Initialization: Constant Model

Before using any features, the model starts with the best **constant prediction**.
Let $\bar{y}$ be the average of the labels:

$$ \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i $$

The optimal initial logit $F_0$ is:

$$ F_0 = \log \left( \frac{\bar{y}}{1 - \bar{y}} \right) $$

This predicts the exact same probability for every sample initially.

---

## 5. Functional Gradient Descent

Gradient Boosting performs gradient descent in **function space**.
At iteration $m$, the model is updated as:

$$ F_m(x) = F_{m-1}(x) + \eta f_m(x) $$

Where:
*   $f_m(x)$ is a **weak learner** (decision stump).
*   $\eta$ is the **learning rate** (step size).

---

## 6. Pseudo-Residuals (Negative Gradients)

To find the best update, we calculate the derivative of the loss function with respect to the prediction $F$:

$$ \frac{\partial L}{\partial F} = p - y $$

Therefore, the **Negative Gradient** (what we want to fit) is:

$$ r_i = y_i - p_i $$

These are called **Pseudo-Residuals**.

**Interpretation:**
*   $r_i > 0$: The label was 1, probability was low ($p < y$). Model **under-predicted**.
*   $r_i < 0$: The label was 0, probability was high ($p > y$). Model **over-predicted**.

---

## 7. Weak Learners: Decision Stumps

Each weak learner is a **Decision Stump** — a regression tree with depth=1 (one split).
It learns a function:

$$ f(x) = \begin{cases} c_L & \text{if } x_j \leq t \\ c_R & \text{if } x_j > t \end{cases} $$

Where:
*   $j$ = selected feature.
*   $t$ = split threshold.
*   $c_L, c_R$ = constant values for left and right leaves.

---

## 8. Why Leaf Values Are Means

Standard Gradient Boosting (for Least Squares) simplifies the leaf calculation. Each leaf predicts a constant value that minimizes the squared error with respect to the **residuals**:

$$ c = \text{argmin}_c \sum (r_i - c)^2 $$

The solution to this optimization is simply the **Mean**:

$$ c = \text{mean}(r_i) $$

*   **Left leaf** predicts mean residual of the left group.
*   **Right leaf** predicts mean residual of the right group.

---

## 9. Training Algorithm (Step-by-Step)

For $m = 1 \dots M$:

1.  **Compute Probabilities:**
   
    $$ p_i = \sigma(F_{m-1}(x_i)) $$

3.  **Compute Residuals:**
   
    $$ r_i = y_i - p_i $$

5.  **Fit a Weak Learner:**
    Train a decision stump on the dataset $\{(x_i, r_i)\}$ to find the best split.

6.  **Update Model:**
   
    $$ F_m(x) = F_{m-1}(x) + \eta f_m(x) $$

Each step reduces the loss greedily.

---

## 10. Prediction Phase

After training, the final model is the sum of the initial value plus all weighted weak learners:

$$ F(x) = F_0 + \sum_{m=1}^{M} \eta f_m(x) $$

To get the classification, we convert this logit sum to a probability:

$$ P(y=1|x) = \sigma(F(x)) $$

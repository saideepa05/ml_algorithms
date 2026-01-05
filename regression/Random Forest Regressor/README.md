# Random Forest Regressor — From Scratch

This folder contains an implementation of **Random Forest Regression** from scratch.

This README explains why Random Forest is needed, how it builds upon Regression Trees using Bagging and Feature Randomness, and the mathematical intuition behind averaging predictions.

## 1. What Problem Random Forest Regression Solves

**A single Regression Tree:**
*   Fits the training data very well.
*   Has **high variance**.
*   **Overfits easily** (memorizes noise).
*   Is unstable (small data changes lead to big model changes).

**Random Forest Regression solves this by:**
*   Training many different regression trees.
*   **Averaging** their predictions.
*   This dramatically improves generalization and stability.

---

## 2. Model Intuition

Instead of trusting one tree, Random Forest asks:

> “What if I train many trees, each seeing a slightly different version of the data, and then average their answers?”

**The Logic:**
*   Each tree is imperfect and makes different mistakes.
*   Averaging them **cancels out errors** and creates a smoother prediction function.

---

## 3. Base Model: Regression Tree (Recap)

Each regression tree predicts a constant value in each leaf:

$$ \hat{y}_{tree}(x) = \text{mean of targets in the leaf containing } x $$

Splits are chosen to minimize **Mean Squared Error (MSE)**.

---

## 4. Random Forest Construction (Step-by-Step)

Let the training data be $\{(x_i, y_i)\}_{i=1}^n$ and the number of trees be $T$.

### Step 1: Bootstrap Sampling (Row Randomness)
For each tree $t = 1, \dots, T$:
Sample $n$ data points **with replacement**. This creates a bootstrap dataset $D^{(t)}$.

**Result:**
*   Some samples appear multiple times.
*   Some samples are missing.
*   **Each tree sees a different dataset.**

### Step 2: Feature Subsampling (Column Randomness)
At **each split** inside a tree:
*   Randomly select a subset of features.
*   Only consider these features for splitting.
*   Typically, if total features is $d$, we select $m = \sqrt{d}$ features.

**Why?** This prevents trees from becoming too similar (decorrelation).

### Step 3: Train Regression Trees
Train each tree independently on its bootstrap dataset using the standard regression tree algorithm.
**Stopping conditions:** Max depth, Min samples per split, Pure node.

---

## 5. Split Criterion (Inside Each Tree)

For a candidate split on feature $j$ with threshold $t$, we define left and right regions:

*   $R_L = \{x : x_j \leq t\}$
*   $R_R = \{x : x_j > t\}$

The cost function is the **Weighted MSE**:

$$ \text{Cost}(j, t) = \frac{|R_L|}{n} \text{Var}(y_L) + \frac{|R_R|}{n} \text{Var}(y_R) $$

The split minimizing this cost is chosen.

---

## 6. Prediction Rule (Forest Output)

Each tree produces a prediction $\hat{y}^{(t)}(x)$.
The Random Forest prediction is simply the **average**:

$$ \hat{y}_{RF}(x) = \frac{1}{T} \sum_{t=1}^{T} \hat{y}^{(t)}(x) $$

---

## 7. Why Averaging Works (Bias–Variance Intuition)

*   **Single Tree:** Low Bias, High Variance.
*   **Random Forest:** Slightly higher Bias, **Much Lower Variance**.

Averaging reduces variance roughly by:

$$ \text{Var}(\hat{y}_{RF}) \propto \frac{1}{T} $$

(Assuming trees are not perfectly correlated).

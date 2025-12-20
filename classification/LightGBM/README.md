# LightGBM — From Scratch (Mathematical Explanation)

This folder contains an implementation of a **LightGBM-style classifier** from scratch using **NumPy**, applied to a binary classification problem (Heart Disease Prediction).

This implementation is not a wrapper around existing libraries. It explicitly implements the core idea that differentiates LightGBM from XGBoost:
*   **Second-order optimization** (Newton boosting).
*   **Regularized leaf weight computation**.
*   **Leaf-wise tree growth strategy**.

## The Goal
**Understand how LightGBM works internally:**
*   How it differs from XGBoost despite using the same loss.
*   Why it grows trees **leaf-wise** instead of level-wise.
*   How gradients and Hessians guide leaf splitting.
*   How controlling the number of leaves prevents overfitting.

---

## 1. What Is LightGBM?

LightGBM (**Light Gradient Boosting Machine**) is a gradient boosting algorithm that builds an ensemble of decision trees sequentially.

Like XGBoost, LightGBM:
1.  Uses **gradients and Hessians**.
2.  Optimizes **logistic loss**.
3.  Uses **regularization**.

However, the key difference is **how trees are grown**.

---

## 2. How LightGBM Differs from XGBoost

| Feature | XGBoost (Level-wise) | LightGBM (Leaf-wise) |
| :--- | :--- | :--- |
| **Growth Strategy** | Grows level-by-level (Depth-first) | Grows leaf-by-leaf (Best-first) |
| **Tree Shape** | Balanced / Symmetric | Unbalanced / Asymmetric |
| **Split Logic** | Splits all nodes at current depth | Splits only the node with max gain |
| **Error Reduction** | Slower per split | Faster per split (Greedy) |

**Leaf-wise strategy** is the defining characteristic of LightGBM. It allows the model to reduce loss faster but requires constraints (`max_leaves`) to prevent overfitting.

---

## 3. What Does the Model Predict?

Like XGBoost, LightGBM does not directly predict class labels. It learns a function $F(x)$ that predicts the **Log-Odds (Logit)**:

$$ F(x) \approx \log \left( \frac{P(y=1|x)}{1 - P(y=1|x)} \right) $$

To get the probability, we apply the **Sigmoid Function**:

$$ P(y=1|x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}} $$

**Final Classification:**

$$ \hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq 0.5 \\ 0 & \text{otherwise} \end{cases} $$

---

## 4. Loss Function (Binary Log-Loss)

LightGBM minimizes the **Binary Cross-Entropy Loss**, identical to XGBoost:

$$ L(y, p) = -[y \log(p) + (1-y) \log(1-p)] $$

Where:
*   $y \in \{0, 1\}$ is the true label.
*   $p = P(y=1|x)$ is the predicted probability.

---

## 5. Second-Order Taylor Approximation

LightGBM uses a **Second-Order Taylor Approximation** of the loss around the current prediction $F$:

$$ L(F+f) \approx L(F) + g f + \frac{1}{2} h f^2 $$

Where:
*   $g = \frac{\partial L}{\partial F}$ (**Gradient** - 1st Derivative)
*   $h = \frac{\partial^2 L}{\partial F^2}$ (**Hessian** - 2nd Derivative)

For **Binary Log-Loss**, these are:

$$ g_i = p_i - y_i $$
$$ h_i = p_i (1 - p_i) $$

These quantities are computed for every sample at every iteration.

---

## 6. Leaf-Based Tree Structure

Instead of growing trees level-wise, LightGBM maintains a **set of leaves**.
*   Each leaf contains a subset of training samples.
*   Each leaf predicts one constant value.

Initially, the entire dataset is assigned to a single root leaf.

---

## 7. Optimal Leaf Weight

For each leaf, LightGBM computes the optimal output value by minimizing the Taylor approximation:

$$ w^* = -\frac{\sum g_i}{\sum h_i + \lambda} $$

Where:
*   $\sum g_i$ is the sum of gradients in the leaf.
*   $\sum h_i$ is the sum of Hessians in the leaf.
*   $\lambda$ is the **L2 Regularization** parameter.

This is identical to XGBoost.

---

## 8. Choosing Which Leaf to Split (Core LightGBM Idea)

At each step, LightGBM performs the following greedy search:

1.  Examines **all** current leaves.
2.  Tries **all** possible feature splits inside each leaf.
3.  Computes the **Gain** for each possible split.
4.  Selects the single split that produces the **maximum gain**.
5.  Splits **only that leaf**.

The **Gain Formula** is:

$$ \text{Gain} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} $$

Where:
*   $G = \sum g_i, \quad H = \sum h_i$ (Total gradient/hessian in the parent leaf).
*   $G_L, H_L$ and $G_R, H_R$ refer to the Left and Right child nodes after the split.

This process ensures that the leaf with the **largest remaining errors** is split first.

---

## 9. Controlling Model Complexity (Max Leaves)

Because leaf-wise growth can produce very deep trees, LightGBM controls complexity using:

*   **Maximum number of leaves (`max_leaves`)**

Once the maximum number of leaves is reached:
1.  Tree growth stops.
2.  The tree is added to the ensemble.

This replaces the `max_depth` parameter primarily used in level-wise trees (though LightGBM can also use `max_depth` as a constraint).

---

## 10. Training Algorithm (Step-by-Step)

**Step 1: Initialization**
Start with a constant prediction (log-odds of the mean):
$$ F_0 = \log \left( \frac{\bar{y}}{1 - \bar{y}} \right) $$

**Step 2: Boosting Loop (for $m = 1 \dots M$)**

1.  **Compute Probabilities:** $p_i = \sigma(F_{m-1}(x_i))$
2.  **Compute Gradients:** $g_i = p_i - y_i$
3.  **Compute Hessians:** $h_i = p_i (1 - p_i)$
4.  **Grow Tree (Leaf-wise):**
    *   Find best leaf to split.
    *   Perform split.
    *   Repeat until `max_leaves` reached.
5.  **Compute Leaf Weights:** $w^* = -\frac{\sum g}{\sum h + \lambda}$
6.  **Update Model:** $F_m(x) = F_{m-1}(x) + \eta f_m(x)$

---

## 11. Prediction Phase

After training, the final model is the sum of all trees:

$$ F(x) = F_0 + \sum_{m=1}^{M} \eta f_m(x) $$

We convert this to a probability for the final result:

$$ P(y=1|x) = \sigma(F(x)) $$

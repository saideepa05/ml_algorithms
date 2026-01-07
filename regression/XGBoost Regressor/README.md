# XGBoost Regressor — Math (From Scratch)

This folder contains an implementation of **XGBoost Regression** from scratch.

This README explains the additive model, the regularized objective function, the **Second-order Taylor Approximation** (the key trick), and the exact formulas used to calculate leaf weights and split gain.

## 1. Additive Model

XGBoost is gradient boosting with trees. The final prediction is a sum of predictions from $T$ trees:

$$ \hat{y}_i = f(x_i) = \sum_{t=1}^{T} \eta h_t(x_i) $$

Where:
*   $h_t$ are regression trees.
*   $\eta$ is the **Learning Rate** (shrinkage).

---

## 2. Objective Function (Regularized)

At boosting round $t$, we add a new tree $h_t$ and minimize the following objective:

$$ L^{(t)} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i^{(t-1)} + h_t(x_i)) + \Omega(h_t) $$

**Regularization Term ($\Omega$):**
This controls the complexity of the tree:

$$ \Omega(h) = \gamma K + \frac{1}{2} \lambda \sum_{j=1}^{K} w_j^2 $$

Where:
*   $K$: Number of leaves.
*   $w_j$: Score (prediction) of leaf $j$.
*   $\gamma$: Penalty per leaf (discourages creating too many leaves).
*   $\lambda$: L2 penalty on leaf weights (controls magnitude).

---

## 3. Second-Order Taylor Approximation

This is the key mathematical trick that makes XGBoost fast and accurate.
We define the **Gradient ($g$)** and **Hessian ($h$)**:

$$ g_i = \frac{\partial \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i} \quad , \quad h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i^2} $$

We approximate the loss function using a second-order Taylor expansion:

$$ \ell(y_i, \hat{y}_i + h_t(x_i)) \approx \ell(y_i, \hat{y}_i) + g_i h_t(x_i) + \frac{1}{2} h_i h_t(x_i)^2 $$

This allows us to fit the tree using only $g$ and $h$, regardless of the specific loss function.

---

## 4. Leaf Weight Formula

For a specific leaf $j$, let $I_j$ be the set of data indices in that leaf.
We define the sums of gradients and hessians in that leaf:

$$ G_j = \sum_{i \in I_j} g_i \quad , \quad H_j = \sum_{i \in I_j} h_i $$

The **Optimal Leaf Value ($w_j^*$)** is calculated as:

$$ w_j^* = -\frac{G_j}{H_j + \lambda} $$

---

## 5. Split Gain Formula

To decide where to split a node, we calculate the **Gain** (Score Improvement).
If a node is split into Left ($L$) and Right ($R$) children:

*   $G = G_L + G_R$
*   $H = H_L + H_R$

The Gain formula is:

$$ \text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma $$

**Decision Rule:** Calculate Gain for all possible splits and choose the one with the **maximum positive gain**.

---

## 6. Squared Error Case (Implementation Detail)

For **Mean Squared Error (MSE)** loss $\ell = \frac{1}{2}(y - \hat{y})^2$:

$$ g_i = \hat{y}_i - y_i $$
$$ h_i = 1 $$

Using these specific values simplifies the calculations while strictly following the XGBoost mathematical framework.

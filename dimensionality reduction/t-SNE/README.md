# t-Distributed Stochastic Neighbor Embedding (t-SNE) — From Scratch (Mathematical Explanation)

This directory contains an implementation of **t-SNE** from scratch using NumPy. t-SNE is a non-linear dimensionality reduction technique particularly well-suited for the visualization of high-dimensional datasets.

## The Goal
Map high-dimensional data into 2D or 3D space such that similar points in high-D remain close to each other in low-D.

---

## 1. High-Dimensional Pairwise Similarities
We start by converting Euclidean distances between data points in high-dimensional space into conditional probabilities that represent similarities.

For two points $x_i$ and $x_j$, the probability $p_{j|i}$ is high if the points are neighbors:
$$ p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)} $$
We then define symmetric probabilities:
$$ p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n} $$

---

## 2. Low-Dimensional Pairwise Similarities
In the low-dimensional space (embedding $y$), we use the **Student t-distribution** with one degree of freedom (which is much heavier-tailed than the Gaussian distribution). This handles the "crowding problem."

$$ q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}} $$

---

## 3. Minimizing KL Divergence
We want the distribution of similarities in low-D ($Q$) to match the distribution in high-D ($P$). We measure the difference using **Kullback-Leibler (KL) Divergence**:

$$ C = KL(P \| Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}} $$

We minimize this cost function using **Gradient Descent**. The gradient of the cost function with respect to the embedding coordinates $y_i$ is:

$$ \frac{\partial C}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1} $$

---

## 4. Algorithm Steps
1.  **Input:** Higher-dim data $X$, Perplexity, Learning Rate.
2.  **Affinity Matrix:** Compute pairwise similarities $p_{ij}$.
3.  **Initialize:** Create initial embedding $Y$ (e.g., random normal).
4.  **Loop:**
    -   Compute low-dim affinities $q_{ij}$.
    -   Compute the gradient $\frac{\partial C}{\partial y}$.
    -   Update embedding: $Y = Y - \eta \cdot \text{gradient}$.
5.  **Output:** 2D or 3D coordinates.

---

## 5. Pros and Cons
*   **Pros:** Outstanding at revealing clusters, captures non-linear structures, excellent for visual exploratory data analysis.
*   **Cons:** Very computationally expensive ($O(N^2)$), stochastic (results change with seed), "global" distances are not preserved (only local neighborhoods matter).

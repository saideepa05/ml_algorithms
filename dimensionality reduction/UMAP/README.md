# Uniform Manifold Approximation and Projection (UMAP) — From Scratch (Mathematical Explanation)

This directory contains a simplified implementation of **UMAP** from scratch using NumPy. UMAP is a modern manifold learning technique that is often faster and preserves more global structure than t-SNE.

## The Goal
Learn the topological structure of a manifold in high-D and project it into a lower-D space while preserving both local and global relationships.

---

## 1. Fuzzy Simplicial Sets
UMAP assumes the data is uniformly distributed on a local manifold. It builds a weighted graph where edge weights represent the probability that two points are connected.

### Local Connectivity
For each point $x_i$, we find its $k$ nearest neighbors. The weight of an edge $(x_i, x_j)$ is:
$$ p_{j|i} = \exp\left( -\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i} \right) $$
where $\rho_i$ is the distance to the nearest neighbor (to ensure connectivity) and $\sigma_i$ is a scaling factor adapted to the local density.

### Symmetrization
We combine weights to form a high-dimensional probability matrix $P$:
$$ p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} p_{i|j} $$

---

## 2. Low-Dimensional Affinities
In the embedding space $Y$, similarities are defined by a curve that looks like a heavy-tailed Student-t distribution but is optimized for faster calculation:
$$ q_{ij} = \left( 1 + a(\|y_i - y_j\|^2)^b \right)^{-1} $$
(Commonly $a=1, b=1$ is used for simplicity).

---

## 3. Optimization via Binary Cross-Entropy
Unlike t-SNE which uses KL divergence, UMAP uses **Binary Cross-Entropy**. This forces the algorithm to not only keep similar points together but also push dissimilar points apart (global structure).

$$ C = \sum_{i \neq j} \left( p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log \frac{1 - p_{ij}}{1 - q_{ij}} \right) $$

---

## 4. Algorithm Steps
1.  **Construct Graph:** Find nearest neighbors and compute high-D probabilities $P$.
2.  **Initialize Embedding:** Usually started with **Spectral Embedding** (from the Graph Laplacian) for stability.
3.  **Optimize:** Use **Stochastic Gradient Descent (SGD)** to minimize the cross-entropy loss.
4.  **Visualize:** Project into 2D or 3D.

---

## 5. Pros and Cons
*   **Pros:** Much faster than t-SNE, preserves global structure better, theoretically grounded in Riemannian geometry.
*   **Cons:** Highly dependent on hyperparameters ($n\_neighbors$, $min\_dist$), results can vary with different initialization metrics.

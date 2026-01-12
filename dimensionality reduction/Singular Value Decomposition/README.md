# Singular Value Decomposition (SVD) — From Scratch (Mathematical Explanation)

This directory contains an implementation of **Singular Value Decomposition (SVD)** from scratch using NumPy. SVD is a fundamental matrix factorization method that generalizes eigendecomposition to any rectangular matrix.

## The Goal
Decompose a data matrix $A$ into three distinct matrices that reveal the underlying geometric structure of the data.

---

## 1. The Matrix Factorization
For any $m \times n$ matrix $A$, SVD states that:

$$ A = U \Sigma V^T $$

Where:
*   $U$: $m \times m$ orthogonal matrix (**Left Singular Vectors**). These represent the relationships between rows (samples).
*   $\Sigma$: $m \times n$ diagonal matrix (**Singular Values**). These represent the strength or "energy" of each component.
*   $V^T$: $n \times n$ orthogonal matrix (**Right Singular Vectors**). These represent the relationships between columns (features).

---

## 2. Relationship to PCA
PCA and SVD are closely related. If we center our data matrix $X$ (so mean = 0), then:
1.  The eigenvectors of $X^T X$ are the **Right Singular Vectors** ($V$).
2.  The eigenvalues $\lambda$ are related to the singular values $\sigma$ by $\lambda = \frac{\sigma^2}{n-1}$.
*Truncated SVD* is often used as a direct replacement for PCA, especially for sparse data.

---

## 3. The Algorithm Steps

### Step 1: Matrix Preparation
Standardize or center the data matrix $X$.

### Step 2: Compute SVD
Perform the decomposition using numerical solvers (like power iteration or NumPy's optimized `linalg.svd`).

### Step 3: Dimensionality Reduction
To reduce the data from $n$ features to $k$ features:
1.  Keep the first $k$ columns of $U$.
2.  Keep the first $k \times k$ block of $\Sigma$.
3.  The reduced data is given by $X_{reduced} = U_k \Sigma_k$.

---

## 4. Why use SVD?
*   **Latent Semantic Analysis:** Finding hidden patterns in text data.
*   **Recommendation Systems:** Matrix completion (e.g., predicted movie ratings).
*   **Image Compression:** Storing only the most significant singular values/vectors.

---

## 5. Pros and Cons
*   **Pros:** Works on non-square matrices, robust to noise, handles multicollinearity perfectly.
*   **Cons:** Computationally expensive for very large matrices, hard to interpret specific components emotionally/physically.

# Independent Component Analysis (ICA) — From Scratch (Mathematical Explanation)

This directory contains an implementation of **Independent Component Analysis (ICA)** from scratch using NumPy. Unlike PCA which looks for uncorrelated components that maximize variance, ICA looks for components that are statistically **independent** and **non-Gaussian**.

## The Goal
To decompose a multivariate signal into additive subcomponents that are maximally independent. This is often used for "Blind Source Separation" (e.g., the Cocktail Party Problem).

---

## 1. Core Assumptions
ICA assumes:
1.  The source signals ($s$) are statistically **independent**.
2.  The source signals have **non-Gaussian** distributions.
3.  The mixing matrix is linear.

$$ x = As $$

where $x$ is the observed data, $A$ is the mixing matrix, and $s$ are the independent sources.

---

## 2. Preprocessing Steps
Before running the main ICA algorithm (like FastICA), we must prepare the data:

### Centering
Subtract the mean to make the data zero-mean.

$$ x = x - E[x] $$

### Whitening
Transform the data so that its components are uncorrelated and have unit variance. This simplifies the search for independent components to a search for a rotation.

$$ \tilde{x} = ExD^{-1/2}E^T x $$

where $E$ is the matrix of eigenvectors of the covariance matrix and $D$ is the diagonal matrix of eigenvalues.

---

## 3. FastICA Algorithm (Fixed-Point Iteration)
The objective is to find a weight vector $w$ such that the projection $y = w^T \tilde{x}$ maximizes **non-Gaussianity**. We measure non-Gaussianity using **Negentropy** approximations.

### The Update Rule
For a single component, the FastICA update step is:
1.  Compute: $w^+ = E[x \cdot g(w^T x)] - E[g'(w^T x)]w$
2.  Normalize: $w = \frac{w^+}{\|w^+\|}$

Common contrast functions $g(u)$:
-   $g(u) = \tanh(u)$
-   $g(u) = u \cdot \exp(-u^2/2)$

---

## 4. Multiple Components (Orthogonalization)
To find $n$ components, we ensure each new vector $w_p$ is orthogonal to the previously found vectors $w_1, \dots, w_{p-1}$:

$$ w_p = w_p - \sum_{j=1}^{p-1} (w_p^T w_j) w_j $$

---

## 5. Pros and Cons
*   **Pros:** Can recover original signals from mixtures, handles non-Gaussian features well.
*   **Cons:** Cannot distinguish the order of components, cannot recover the exact scale/amplitude of sources, fails if sources are Gaussian.

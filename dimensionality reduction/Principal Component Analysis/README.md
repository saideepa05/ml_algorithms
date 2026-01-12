# Principal Component Analysis (PCA) — From Scratch (Mathematical Explanation)

This directory contains an implementation of **Principal Component Analysis (PCA)** from scratch using NumPy. PCA is the most popular dimensionality reduction technique, used to simplify data while preserving as much variance as possible.

## The Goal
Find a new set of orthogonal axes (Principal Components) that represent the directions of maximum variance in the data.

---

## 1. Mathematical Intuition
PCA transforms a set of $d$ correlated variables into a smaller set of $k$ uncorrelated variables. It does this by finding the eigenvectors of the data's covariance matrix.

---

## 2. The Step-by-Step Math

### Step 1: Standardize the Data

$X = \frac{X - \mu}{\sigma}$ 

Note: In the implementation, we focus on centering ($X - \mu$).

### Step 2: Compute the Covariance Matrix
The covariance matrix $C$ represents how features vary together:

$$ C = \frac{1}{n-1} X^T X $$

### Step 3: Eigen-Decomposition
We find the eigenvalues ($\lambda$) and eigenvectors ($v$) that satisfy:

$$ Cv = \lambda v $$

*   **Eigenvalues:** Represent the amount of variance captured by each principal component.
*   **Eigenvectors:** Define the direction of the new axes.

### Step 4: Sort and Select
Sort eigenvectors by their eigenvalues in descending order. Pick the top $k$ eigenvectors to form a projection matrix $W$.

### Step 5: Projection
Project the original data onto the new subspace:

$$ X_{new} = X \cdot W $$

---

## 3. Explained Variance
To determine how many components to keep, we calculate the **Explained Variance Ratio**:

$$ \text{Ratio}_i = \frac{\lambda_i}{\sum \lambda} $$

---

## 4. Pros and Cons
*   **Pros:** Removes correlated features, reduces noise, makes visualization of high-D data possible, speeds up ML models.
*   **Cons:** Linear only (cannot capture complex curves), Principal Components are hard to interpret (they are mixes of raw features), sensitive to outliers.

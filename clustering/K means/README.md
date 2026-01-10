# K-Means Clustering — From Scratch (Mathematical Explanation)

This directory contains an implementation of the K-Means Clustering algorithm from scratch using **NumPy**.

This README explains the intuition, mathematics, and learning process behind K-Means without using any external ML libraries.

## The Goal
**Partition the dataframe into $K$ distinct, non-overlapping subgroups (clusters).** Each data point belongs to the cluster with the nearest mean (centroid).

---

## 1. What K-Means Does

K-Means is an unsupervised learning algorithm. Given a dataset of points, it tries to find $K$ centers (centroids) and assigns every point to the closest center.

It minimizes the variance within each cluster.

---

## 2. The Objective Function (Inertia)

The "goodness" of a clustering is measured by **Inertia** (or Within-Cluster Sum of Squares). We want to **minimize** this value.

$$ \text{Inertia} = \sum_{i=1}^{K} \sum_{x \in C_i} || x - \mu_i ||^2 $$

Where:
*   $K$: Number of clusters.
*   $x$: A single data point.
*   $C_i$: The $i$-th cluster.
*   $\mu_i$: The centroid (mean) of cluster $C_i$.
*   $|| x - \mu_i ||^2$: The squared Euclidean distance between point $x$ and centroid $\mu_i$.

**Lower inertia = tighter clusters.**

---

## 3. The Algorithm Steps

K-Means is an iterative algorithm. It repeats two main steps until "convergence" (when centroids stop moving).

### Step 1: Initialization
Randomly select $K$ data points from the dataset to serve as the initial centroids:
$$ \mu_1, \mu_2, ..., \mu_K $$

### Step 2: Assignment (Expectation Step)
For every data point $x$, calculate its distance to every centroid. Assign $x$ to the cluster $C_i$ whose centroid $\mu_i$ is closest.

$$ \text{Cluster}(x) = \text{argmin}_{i} (|| x - \mu_i ||^2) $$

### Step 3: Update (Maximization Step)
Recalculate the centroids. The new centroid $\mu_i$ is simply the **mean** of all points assigned to cluster $C_i$.

$$ \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x $$

### Step 4: Iteration
Repeat Steps 2 and 3 until the centroids do not change significantly (convergence) or a maximum number of iterations is reached.

---

## 4. Choosing the Number of Clusters ($K$)

Since $K$ is a hyperparameter, how do we pick it?

### The Elbow Method
We run K-Means for a range of $K$ values (e.g., 1 to 10) and calculate the **Inertia** for each.
*   As $K$ increases, Inertia *always* decreases.
*   We look for the "elbow point" where the rate of decrease dramatically slows down. This is usually the optimal $K$.

---

## 5. Pros and Cons

*   **Pros:** Simple, fast, scales well to large datasets.
*   **Cons:** Must specify $K$ manually, sensitive to initialization (can get stuck in local optima), assumes clusters are spherical/convex.

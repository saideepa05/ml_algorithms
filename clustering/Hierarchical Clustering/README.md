# Hierarchical Clustering — From Scratch (Mathematical Explanation)

This directory contains an implementation of Agglomerative Hierarchical Clustering from scratch using **NumPy** and **SciPy** (for dendrogram visualization).

This README explains the intuition, mathematics, and learning process behind Hierarchical Clustering without using any external ML libraries for the core logic.

## The Goal
**Build a hierarchy of clusters** by iteratively merging the closest pairs of clusters (Agglomerative) or splitting the farthest ones (Divisive). We focus on the Agglomerative (bottom-up) approach.

---

## 1. What Hierarchical Clustering Does

Unlike K-Means, Hierarchical Clustering does not require specifying the number of clusters $K$ effectively. Instead, it produces a **Dendrogram** (a tree diagram) that shows how clusters are merged at different distance thresholds. You can "cut" the tree at any level to decide $K$.

---

## 2. Measuring Distance (Linkage)

The core decision in this algorithm is: **"Which two clusters are closest?"**

To measure the distance between two clusters (sets of points) $A$ and $B$, we use a **Linkage Metric**.

### Single Linkage (Min Distance)
Distance between the *closest* pair of points in the two clusters.

$$ D(A, B) = \min \{ d(a, b) : a \in A, b \in B \} $$

*   *Effect:* Can result in long, "chain-like" clusters.

### Complete Linkage (Max Distance)
Distance between the *farthest* pair of points in the two clusters.

$$ D(A, B) = \max \{ d(a, b) : a \in A, b \in B \} $$

*   *Effect:* Tends to find compact, spherical clusters.

### Average Linkage
Average distance between all pairs of points in the two clusters.

$$ D(A, B) = \frac{1}{|A| \cdot |B|} \sum_{a \in A} \sum_{b \in B} d(a, b) $$

### Ward's Method
Minimizes the increase in total **within-cluster variance** after merging. This is similar to the objective function of K-Means and often produces the most cohesive clusters.

---

## 3. The Algorithm Steps (Agglomerative)

### Step 1: Initialization
Start with $N$ clusters, where every data point is its own cluster.

$$ \{x_1\}, \{x_2\}, ..., \{x_N\} $$

### Step 2: Distance Matrix
Calculate the pairwise distance between all active clusters using the chosen Linkage metric.

### Step 3: Merge
Find the pair of clusters $(C_i, C_j)$ with the **minimum distance**. Merge them into a new single cluster.

$$ C_{new} = C_i \cup C_j $$

### Step 4: Iteration
Repeat Steps 2 and 3. In each step, the number of clusters decreases by 1 ($N \rightarrow N-1 \rightarrow ... \rightarrow 1$).
Stop when only one single cluster containing all points remains.

---

## 4. The Dendrogram

The result is visualized as a Dendrogram.
*   **X-axis:** The data points (or clusters).
*   **Y-axis:** The distance (height) at which a merge occurred.

To find $K$ clusters, we draw a horizontal line across the dendrogram. The number of vertical lines it intersects is the number of clusters.

---

## 5. Pros and Cons

*   **Pros:** No need to pre-specify $K$, hierarchical structure provides rich insights, good for small datasets.
*   **Cons:** Computationally expensive ($O(N^3)$ or $O(N^2)$), sensitive to noise/outliers, hard to undo splits/merges.

# DBSCAN — From Scratch (Mathematical Explanation)

This directory contains an implementation of **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) from scratch using **NumPy**.

This README explains the intuition, mathematics, and learning process behind DBSCAN without using any external ML libraries.

## The Goal
**Cluster data based on density.** Unlike K-Means, DBSCAN groups together points that are close to each other in high-density regions and marks points in low-density regions as **outliers (noise)**. It can find clusters of arbitrary shapes.

---

## 1. Parameters

DBSCAN uses two main parameters to define "density":

1.  **Epsilon ($\epsilon$):** The radius around a point.
2.  **MinPts (Minimum Points):** The minimum number of points required within that radius to consider the region "dense".

---

## 2. Core Concepts (Point Classification)

For every point in the dataset, we classify it into one of three categories:

### A. Core Point
A point $p$ is a **Core Point** if there are at least $MinPts$ within its $\epsilon$-neighborhood (including itself).
$$ |N_\epsilon(p)| \ge MinPts $$
where $N_\epsilon(p) = \{ q \in D \mid \text{dist}(p, q) \le \epsilon \}$.

### B. Border Point
A point that is **not** a Core Point, but it falls within the $\epsilon$-neighborhood of a Core Point. It is on the edge of a cluster.

### C. Noise Point (Outlier)
A point that is neither a Core Point nor a Border Point. It belongs to no cluster.

---

## 3. The Algorithm Steps

DBSCAN classifies points and builds clusters by "crawling" through the dense connections.

### Step 1: Visit Points
Pick an arbitrary unvisited point $p$ from the dataset.

### Step 2: Check Neighborhood
Retrieve all points in the $\epsilon$-neighborhood of $p$.

### Step 3: Expand Cluster or Mark Noise
*   **Case 1:** If $|N_\epsilon(p)| < MinPts$:
    *   Mark $p$ as **Noise** (temporarily). It might be found later as a Border Point of a different cluster.
    *   Move to the next unvisited point.
*   **Case 2:** If $|N_\epsilon(p)| \ge MinPts$:
    *   $p$ is a Core Point. Creates a **new cluster**.
    *   Add $p$ and all its neighbors to the cluster.
    *   **Recursively** check the neighbors of each neighbor. If a neighbor is also a Core Point, expand the cluster to include *its* neighbors. This chain reaction continues until the cluster is fully expanded (all density-connected points are found).

### Step 4: Iteration
Continue steps 1-3 until every point in the dataset has been visited and labeled.

---

## 4. Pros and Cons

*   **Pros:**
    *   Does not require specifying $K$ (number of clusters).
    *   Can find arbitrarily shaped clusters (e.g., crescents, rings) that K-Means fails on.
    *   Robust to outliers (isolates noise).
*   **Cons:**
    *   Cannot handle varying densities well (one $\epsilon$ fits all).
    *   Sensitive to parameters $\epsilon$ and $MinPts$.
    *   Distance calculation is computationally expensive ($O(N^2)$ without spatial indexing).

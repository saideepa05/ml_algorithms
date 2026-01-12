# Gaussian Mixture Models (GMM) — From Scratch (Mathematical Explanation)

This directory contains an implementation of Gaussian Mixture Models from scratch using **NumPy**.

This README explains the intuition, mathematics, and learning process behind GMM without using any external ML libraries.

## The Goal
**Model the data as a mixture of finite Gaussian distributions.** Unlike K-Means, which performs "hard assignment" (a point belongs to one cluster), GMM performs **"soft assignment"** (a point belongs to a cluster with a certain probability).

---

## 1. What GMM Does

GMM assumes that the dataset is generated from a mixture of $K$ different Gaussian distributions. The goal is to find the defining parameters of these Gaussians:
1.  **Mean ($\mu$):** The center of the cluster.
2.  **Covariance ($\Sigma$):** The spread and shape of the cluster (circle, oval, etc.).
3.  **Mixing Coefficient ($\pi$):** The weight or size of the cluster (probability that any random point belongs to it).

---

## 2. The Gaussian Distribution

For a single data point $x$ in $D$-dimensions, the probability density function (PDF) for a specific Gaussian component $k$ is:

$$ \mathcal{N}(x | \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{D/2} |\Sigma_k|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right) $$

The total probability of observing a point $x$ is the weighted sum of all $K$ Gaussians:

$$ p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) $$

Where $\sum \pi_k = 1$.

---

## 3. The Algorithm (Expectation-Maximization)

Since we don't know the parameters ($\mu, \Sigma, \pi$) AND we don't know which point belongs to which cluster (latent variables), we use the **Expectation-Maximization (EM)** algorithm.

### Step 1: Initialization
Randomly initialize the parameters:
*   Means $\mu_k$ (e.g., using random points).
*   Covariances $\Sigma_k$ (e.g., identity matrices).
*   Mixing coefficients $\pi_k$ (e.g., uniform $1/K$).

### Step 2: E-Step (Expectation)
Calculate the **Responsibility** $\gamma(z_{nk})$. This is the posterior probability that point $x_n$ belongs to cluster $k$.

$$ \gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n|\mu_j, \Sigma_j)} $$

*   "How much does Gaussian $k$ claim this point?"

### Step 3: M-Step (Maximization)
Update the parameters using the responsibilities calculated in the E-Step.

*   **New Means:** Weighted average of points.
 
    $$ \mu_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) x_n $$
    
*   **New Covariances:** Weighted variance.
 
    $$ \Sigma_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (x_n - \mu_k^{new})(x_n - \mu_k^{new})^T $$
    
*   **New Mixing Coefficients:** Fraction of total weight.

    $$ \pi_k^{new} = \frac{N_k}{N} $$

Where $N_k = \sum_{n=1}^{N} \gamma(z_{nk})$.

### Step 4: Iteration
Repeat the E-Step and M-Step. In each iteration, we calculate the **Log-Likelihood** of the data. We visualize/stop when the Log-Likelihood stabilizes (convergence).

$$ \ln p(X | \pi, \mu, \Sigma) = \sum_{n=1}^{N} \ln \left\{ \sum_{k=1}^{K} \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k) \right\} $$

---

## 4. Pros and Cons

*   **Pros:** Soft clustering (probabilities), flexible cluster shapes (ellipsoidal), handles overlapping clusters well.
*   **Cons:** Slower than K-Means, sensitive to initialization, can converge to local optima, requires specifying $K$.

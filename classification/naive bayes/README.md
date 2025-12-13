# Naive Bayes — From Scratch (Mathematical Explanation)

This document explains **Naive Bayes Classification** from scratch using pure **NumPy**.
It covers both:

 **Gaussian Naive Bayes** — for continuous features
 **Categorical Naive Bayes** — for categorical/count features

Naive Bayes is one of the simplest yet most powerful classification algorithms, based on **Bayes’ Theorem** and the assumption that features are independent given the class.

---

## 1. What Is Naive Bayes?

Naive Bayes predicts the class of a new item by asking:

> “Which class makes this data most likely?”

It uses probabilities, not distances or optimization.

Given features $x = (x_1, x_2, ..., x_d)$, we want to compute:

$$ P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)} $$

Since $P(x)$ is the same for all classes, we only compare:

$$ P(y|x) \propto P(y) \cdot P(x|y) $$

Under the **Naive (independence) assumption**:

$$ P(x|y) = \prod_{i=1}^{d} P(x_i|y) $$

---

## 2. Step-by-Step: How Naive Bayes Works

To classify a new point, we follow these steps:

### Step 1: Compute the Prior Probability
This is simply the frequency of each class in the training data:

$$ P(y) = \frac{\text{count of class } y}{\text{total samples}} $$

### Step 2: Compute the Likelihood for Each Feature
The method depends on the data type.

#### ● For Continuous Features → Gaussian Naive Bayes
We assume each feature follows a **Normal (Gaussian) Distribution**:

$$ P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp \left( - \frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2} \right) $$

We estimate parameters from training data:
*   **Mean ($\mu_{y,i}$):** Average of feature $i$ in class $y$.
*   **Variance ($\sigma_{y,i}^2$):** Variance of feature $i$ in class $y$.

#### ● For Categorical Features → Multinomial/Bernoulli Naive Bayes

**Multinomial NB** (Used for word counts, frequencies):
Uses Laplace smoothing (adding +1) to handle zero probabilities.

$$ P(x_i|y) = \frac{N_{i,y} + 1}{\sum_j N_{j,y} + K} $$

**Bernoulli NB** (Used for binary yes/no features):

$$ P(x_i=1|y) = \theta_{y,i} $$
$$ P(x_i=0|y) = 1 - \theta_{y,i} $$

---

## 3. Final Prediction Rule

Instead of multiplying many small probabilities (which causes numerical underflow), we sum their **Logs**.

We compute the posterior probability and choose the maximum:

$$ \hat{y} = \text{argmax}_y \left[ \log P(y) + \sum_{i=1}^{d} \log P(x_i|y) \right] $$

**Why Log?**
1.  Turns multiplication into addition.
2.  Prevents computer errors with tiny numbers.
3.  Preserves the order (the maximum value remains the maximum).

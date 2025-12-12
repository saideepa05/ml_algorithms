# K-Nearest Neighbors (KNN) — From Scratch

This document explains **K-Nearest Neighbors (KNN)** from scratch using NumPy.  
KNN is one of the simplest and most intuitive machine learning algorithms — yet extremely powerful for classification tasks.

Unlike Logistic Regression, **KNN does not learn weights**.  
Instead, it makes predictions by simply comparing a new sample to the existing training data.

---

## 1. What KNN Does

KNN is a **classification algorithm** that predicts the label of a new data point by looking at the **K most similar examples** from the training dataset.

Example (Heart Disease Prediction):

> “If most of the K closest patients had heart disease,  
> the new patient probably has it too.”

KNN is based entirely on **similarity (distance)** — not on learning parameters.

---

## 2. The Core Idea

To classify a new point:

1. Compute the distance between the new point and **every** point in the training set.  
2. Pick the **K closest points** (neighbors).  
3. Look at their labels.  
4. Choose the **majority label**.

Example (K = 5):


---

## 3. Distance Calculation (Euclidean Distance)

Given two points:

- \( x = (x_1, x_2, ..., x_d) \)
- \( y = (y_1, y_2, ..., y_d) \)

The Euclidean distance is:

$$
d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \dots + (x_d - y_d)^2}
$$

This tells us:

**Smaller distance → more similar points.**

---

## 4. Making a Prediction

Given a new sample \( x \):

### **Step 1 — Compute distances**

$$
d_i = d(x, x^{(i)})
$$

for all training samples.

### **Step 2 — Pick the K nearest neighbors**

Sort distances → choose the indices of the smallest K values.

### **Step 3 — Majority vote**

If most neighbors are labeled **1**, predict **1**.  
If most neighbors are labeled **0**, predict **0**.

Mathematically:

$$
\hat{y} = \text{mode}(y^{(1)}, y^{(2)}, ..., y^{(k)})
$$

---

## 5. Why KNN is a “Lazy Learner”

KNN performs **no training**.

It simply:

- stores the training data  
- waits for a query  
- computes distances at prediction time  

This is why KNN is often called a **lazy algorithm** — it learns *only when needed*.

---

## 6. Choosing the Value of K

### **Small K (1, 3):**
- Very sensitive to noise  
- High variance  
- Can overfit  

### **Large K (15, 25):**
- Smoother decision boundaries  
- May oversimplify patterns  




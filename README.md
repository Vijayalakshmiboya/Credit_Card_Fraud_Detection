# Credit_Card_Fraud_Detection

# Support Vector Machines (SVM)

# 📌 What are SVMs?

A __Support Vector Machine (SVM)__ is a supervised machine learning algorithm used for __classification and regression tasks__.
It works by finding the __optimal hyperplane__ that separates classes in an N-dimensional space while __maximizing the margin__ (distance between support vectors and the hyperplane).

* If the data has __2 features → line__ separates the classes.
* If the data has __3 features → plane__ separates the classes.
* If the data has __n features → hyperplane__ is formed.
<img width="4845" height="2807" alt="image" src="https://github.com/user-attachments/assets/5dfc8112-b976-42eb-8b4b-147406229416" />




👉 The closest points to this hyperplane are called __support vectors__. These points define the decision boundary.

SVMs were introduced by __Vladimir Vapnik and colleagues__ in the 1990s and became a popular choice for classification problems in text, image, and bioinformatics.

---

# 🛠 Types of SVM Classifiers

# 1. Linear SVM

* Works when the data is __linearly separable__.

* Tries to “fit the widest street” between two classes.

* Equation of hyperplane:
  
    **w.x+b=0**


* __Hard Margin SVM__ → strict separation, no errors allowed.

* __Soft Margin SVM__ → allows misclassifications (controlled by hyperparameter **C**).

---

# 2. Nonlinear SVM

Most real-world data is __not linearly separable__. To handle this, SVM uses the __Kernel Trick__.

* Kernels map the data into __higher-dimensional space__ to make it linearly separable.
* Common Kernels:

  ## ⚙️ Common SVM Kernels

    ### 🔹 Linear Kernel
    - Simplest kernel, no transformation applied.
    - Best when data is **linearly separable** (a straight line or hyperplane can separate classes).
    - Equation:  
      K(x, y) = x · y
          
    ### 🔹 Polynomial Kernel
    - Maps data into a **higher-dimensional space** using polynomial features.
    - Useful when data is not linearly separable but has polynomial relationships.
    - Equation:  
      K(x, y) = (x · y + c)^d  
      (where `d` = degree of polynomial, `c` = constant)
    
    ### 🔹 Radial Basis Function (RBF) / Gaussian Kernel
    - Most widely used kernel.
    - Creates **nonlinear decision boundaries** by considering the similarity between points.
    - Great for complex data where classes are not linearly separable.
    - Equation:  
      K(x, y) = exp(-γ ||x − y||²)  
      (γ controls influence of single training point)
    
    ### 🔹 Sigmoid Kernel
    - Based on the **sigmoid function**, similar to activation in neural networks.
    - Works well in some cases but less popular than RBF.
    - Equation:  
      K(x, y) = tanh(α (x · y) + c)  
      (α and c are constants)

# 3. Support Vector Regression (SVR)

* Extension of SVM for **regression tasks**.
* Finds a hyperplane within a **tolerance margin (ε)**.
* Useful in **time series prediction, stock forecasting, etc.**
  

# ⚙️ Key Hyperparameters in SVM

1. # C (Regularization parameter)

   * Controls the __trade-off between margin size and classification error__.
   * Small C → wider margin, more tolerance for misclassification.
   * Large C → narrower margin, fewer misclassifications.

2. # Gamma (γ in RBF Kernel)**

   * Controls __influence of a single training example__.
   * Low γ → far influence (smoother boundary).
   * High γ → close influence (tighter boundary, risk of overfitting).

3. # Kernel Type

   * Defines how data is transformed into higher dimensions.
   * Choice depends on dataset structure.


# 🔍 Hyperparameter Tuning Approaches

1. # Plain SVM (default parameters)

   * Fast but rarely optimal.
   * Only good for baseline comparison.

2. # Grid Search with Cross-Validation (GridSearchCV)

   * Exhaustively tries all parameter combinations.
   * Guarantees best parameters but __computationally expensive__.

3. # Randomized Search CV 

   * Randomly samples parameter combinations.
   * Faster than grid search, good for large parameter spaces.
   * May not always find the absolute best parameters.

# ✅ Applications of SVM 
While SVMs can be applied for a number of tasks, these are some of the most popular applications of SVMs across industries.

**Text classification**
SVMs are commonly used in natural language processing (NLP) for tasks such as sentiment analysis, spam detection, and topic modeling. 
They lend themselves to these data as they perform well with high-dimensional data.

**Image classification**
SVMs are applied in image classification tasks such as object detection and image retrieval. It can also be useful in security domains, 
classifying an image as one that has been tampered with.

**Bioinformatics**
SVMs are also used for protein classification, gene expression analysis, and disease diagnosis. SVMs are often applied in cancer research 
(link resides outside ibm.com) because they can detect subtle trends in complex datasets.

**Geographic information system (GIS)**
SVMs can analyze layered geophysical structures underground, filtering out the 'noise' from electromagnetic data. They have also helped to predict 
the seismic liquefaction potential of soil, which is relevant to field of civil engineering.


# ✅ Advantages of SVM

1. Works well in high-dimensional spaces (text, images).
2. Effective for nonlinear classification using kernel trick.
3. Robust against overfitting if parameters are tuned properly.



# ❌ Limitations of SVM

1. Computationally expensive for large datasets.
2. Choice of kernel & hyperparameters is critical.
3. Hard to interpret compared to models like decision trees.


## 🔍 Hyperparameter Tuning in SVM

### 1️⃣ Plain SVM (default parameters)
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plain SVM (default parameters)
svm = SVC()
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("Plain SVM Results:")
print(classification_report(y_test, y_pred))

# Module Topic Usage in Machine Learning

<h2>Liner Algebra </h2>

## Scalars

**What are Scalars**  
A scalar is a single numeric quantity, fundamental in machine learning for computations, and deep learning for things like learning rates and loss values.  
**Importance**: Important

## Vectors
**What are Vectors**  
These are arrays of numbers that can represent multiple forms of data. In machine learning, vectors can represent data points; in deep learning, they can represent features, weights, and biases.  
**Importance**: Very Important

## Row Vector and Column Vector
Different forms of representing vectors. These representations affect computations like matrix multiplication, critical in neural network operations.

## Distance from Origin  
The magnitude of a vector from the origin. Important for normalization in ML and understanding weights in DL.  
**Importance**: [L] Later

## Euclidean Distance between Two Vectors  
Measures the straight-line distance between vectors. Common in clustering, nearest neighbor search, and MSE loss functions.

## Scalar Vector Addition/Subtraction (Shifting)  
Useful for data normalization and centering; also used in bias correction.

## Scalar Vector Multiplication/Division (Scaling)  
Used for data scaling and controlling learning rates.

## Vector Vector Addition/Subtraction  
Combines or compares vectors; used in computations on data and weights.

## Dot Product of Two Vectors  
Results in a scalar; used for similarity, weighted sums in neural nets.

## Angle between Two Vectors  
Indicates directional difference; used in recommendation systems and understanding high-dimensional relationships.

## Unit Vectors  
Important for normalization and consistent direction in weight updates.

## Projection of a Vector  
Used in dimensionality reduction and visualizing features.

## Basis Vectors  
Help define coordinate systems and understand PCA/SVD transformations.

## Equation of a Line in n-D  
Used in linear regression and in defining hyperplanes in deep learning.

## Vector Norms  
Measure the length of vectors. Used in regularization and normalization.  
**Importance**: [L]

## Linear Independence  
Critical for regression and PCA. Collinearity affects model stability.

## Vector Spaces  
Concept used in feature/output spaces in ML/DL, and in clustering and transformations in neural networks.

---

# Matrix

## What are Matrices?  
Two-dimensional arrays used to represent features, parameters, or transformations.

## Types of Matrices  
Identity, zero, sparse, etc., used in linear algebra and large-scale data operations.

## Orthogonal Matrices  
Preserve length/angle, useful in PCA/SVD and weight initialization.

## Symmetric Matrices  
Equal to their transpose. Common in covariance matrices.

## Diagonal Matrices  
Used for scaling and in optimization schedules.

## Matrix Equality  
Fundamental check in algorithm convergence.

## Scalar Operations on Matrices  
Used to scale or adjust matrix values.

## Matrix Addition and Subtraction  
Used to manipulate datasets or model parameters.

## Matrix Multiplication  
Core operation in regression, forward/backpropagation.

## Transpose of a Matrix  
Important for dot product and other matrix operations.

## Determinant  
Used in multivariate distributions and advanced model architectures.

## Minor and Cofactor  
Important in calculating inverses and understanding determinants.

## Adjoint of a Matrix  
Used in inverse calculations for systems of equations.

## Inverse of a Matrix  
Used in solving systems and pseudo-inverse applications in DL.

## Rank of a Matrix  
Used to determine solvability and matrix properties.

## Column Space and Null Space  
Describe linear system solutions, especially in regression.  
**Importance**: [L]

## Change of Basis  
Used in transforming between coordinate systems; important in PCA.  
**Importance**: [L]

## Solving a System of Linear Equations  
Core to regression and optimization in DL.

## Linear Transformations  
Fundamental mapping concept across ML and DL.

## 3D Linear Transformations  
Used in geometric interpretations and visualization.

## Matrix Multiplication as Composition  
Neural network layers often represent composed transformations.

## Linear Transformation of Non-square Matrix  
Used for dimensionality reduction and feature construction.

## Dot Product  
Scalar result from vector multiplication; used in similarity and NN layers.

## Cross Product  
Results in orthogonal vector; occasionally used in 3D ML problems.  
**Importance**: [L]

---

# Tensors

## What are Tensors?  
Generalization of scalars, vectors, matrices. Represent data across dimensions.

## Importance of Tensors in Deep Learning  
They are the core data structures in DL frameworks like TensorFlow and PyTorch.

## Tensor Operations  
Include addition, multiplication, reshaping—used in all DL models.

## Data Representation using Tensors  
Example: Image = 3D Tensor (Height × Width × Channels)

---

# Eigen Values and Vectors

## Eigen Vectors and Eigen Values  
Used in PCA and understanding transformations.

## Eigen Faces  
Application of eigenvectors in facial recognition.  
**Importance**: [L]

## Principal Component Analysis (PCA)  
Dimensionality reduction method used for visualization and noise removal.  
**Importance**: [L]

---

# Matrix Factorization

## LU Decomposition  
Solves linear systems; used in regression.  
**Importance**: [L]

## QR Decomposition  
Used for numerical stability and solving equations.  
**Importance**: [L]

## Eigen Decomposition  
Used for structural data analysis and PCA.  
**Importance**: [L]

## Singular Value Decomposition (SVD)  
Used for dimensionality reduction, compression, and recommendation systems.  
**Importance**: [L]


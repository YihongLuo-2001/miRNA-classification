import joblib
import numpy as np


# Function to calculate the polynomial decision function for a given input x
def polynomial_kernel_function(x, degree, coef0):
    return (np.dot(support_vectors, x.T) + coef0) ** degree


# Function to print the hyperplane equation
def hyperplane_function(w, b):
    equation = " + ".join([f"{w[i]:.3f}*x{i + 1}" for i in range(len(w))])
    equation += f" + {b:.3f} = 0"
    return equation


# model = joblib.load('../families_20/svm_model_6_families.joblib')
model = joblib.load('../../models_results/animal_plant/svm_model_6.joblib')

# Check if the kernel is 'poly'
if model.kernel != 'poly':
    if model.kernel == 'linear':
        # Get the coefficients (weights) and intercept (bias) of the hyperplane
        w = model.coef_[0]  # Weights (coefficients of the hyperplane)
        b = model.intercept_[0]  # Intercept (bias term)

        # Print the hyperplane equation
        print("The hyperplane function is:")
        print(hyperplane_function(w, b))
        exit()
    else:
        raise ValueError(
            "This program is designed to output the hyperplane for an SVM with a polynomial or linear kernel.")

# Get the degree, coefficients (weights), and intercept (bias) of the polynomial kernel
degree = model.degree  # Degree of the polynomial kernel
coef0 = model.coef0  # Independent term in the polynomial kernel
support_vectors = model.support_vectors_  # Support vectors
dual_coef = model.dual_coef_  # Coefficients of the support vectors in the decision function
intercept = model.intercept_  # Intercept term

# Print information about the polynomial kernel model
print(f"Polynomial kernel of degree: {degree}")
print(f"Coefficient (coef0): {coef0}")
print(f"Intercept: {intercept}")

# Example: Adjust sample input to match the support vector shape (136 features in this case)
sample_input = np.random.rand(1, support_vectors.shape[1])  # Example input with 136 features

# Compute the polynomial kernel decision function
decision_function = np.dot(dual_coef, polynomial_kernel_function(sample_input, degree, coef0)) + intercept
print(f"Decision function value for input {sample_input}: {decision_function}")

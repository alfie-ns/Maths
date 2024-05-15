#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# --------------------------------------------------------------

'''
    This function generates synthetic linear data with noise. The goal is to create a dataset 
    where the true relationship between the input variable X and the output variable y 
    follows a linear model with added Gaussian noise.

    The linear relationship is defined as:
        y = 4 + 3X + noise

    Here:
    - `X` is the input variable and is randomly generated within the range [0, 2).
    - `y` is the output variable which is linearly dependent on `X` plus some random noise.
    - The noise is added to simulate real-world data where measurements and observations 
      are not perfect and contain some randomness.

    Mathematically:
    1. `X = 2 * np.random.rand(n_samples, 1)` generates n_samples random values for X.
       - `np.random.rand(n_samples, 1)` creates a column vector with n_samples random values 
         uniformly distributed in the range [0, 1).
       - Multiplying by 2 scales these values to the range [0, 2).

    2. `y = 4 + 3 * X + np.random.randn(n_samples, 1)` computes the corresponding y values.
       - `4` is the intercept term (the value of y when X = 0).
       - `3` is the slope of the line (how much y increases for a unit increase in X).
       - `np.random.randn(n_samples, 1)` adds Gaussian noise with mean 0 and standard deviation 1 
         to each y value. This noise term is what makes the data realistic, simulating the 
         variability you would expect in real-world data.

    The function returns the generated X and y values as NumPy arrays.
'''

def generate_linear_data(n_samples=100):
    """Generate synthetic linear data with noise."""
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    return X, y

# --------------------------------------------------------------

def generate_logistic_data(n_samples=100):
    """
        This function generates logisitic data
    
    
    
    
    
    """
    np.random.seed(42) # pass in a random seed(42)
    X = 3 * np.random.rand(n_samples, 1) - 1.5
    y = (X > 0).astype(int).ravel()
    return X, y

# --------------------------------------------------------------

def compute_loss(X, y, theta):
    # Compute the mean squared error loss.(MSE)
    m = len(y) # Number of samples: m=
    predictions = X.dot(theta)
    loss = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return loss

# --------------------------------------------------------------

def gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=1000):
    """Perform gradient descent to learn theta."""
    m = len(y)
    loss_history = np.zeros(n_iterations)

    for i in range(n_iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
        loss_history[i] = compute_loss(X, y, theta)

    return theta, loss_history

# --------------------------------------------------------------

def polynomial_regression(X, y, degree=2):
    """Train a polynomial regression model."""
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    theta = np.linalg.pinv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    return theta, poly_features

# --------------------------------------------------------------

def ridge_regression(X, y, alpha=1.0):
    """Train a ridge regression model."""
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

# --------------------------------------------------------------

def logistic_regression(X, y):
    """Train a logistic regression model."""
    model = LogisticRegression()
    model.fit(X, y)
    return model

# --------------------------------------------------------------

def add_explanation_below(ax, text):
    """Add explanation text below the plot."""
    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.3)  # Adjust the bottom margin to make space for text
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text((xlim[1] - xlim[0]) / 2 + xlim[0], ylim[0] - 0.1 * (ylim[1] - ylim[0]), text,
            ha='center', va='top', fontsize=10, wrap=True, bbox=dict(facecolor='white', alpha=0.5))
    
# --------------------------------------------------------------

def train_models():
    """Train various regression models and demonstrate their performance."""
    # Linear Regression
    X, y = generate_linear_data()
    X_b = np.c_[np.ones((len(X), 1)), X]  # Add bias term (x0 = 1)
    theta = np.random.randn(2, 1)
    learning_rate = 0.01
    n_iterations = 1000
    theta, loss_history = gradient_descent(X_b, y, theta, learning_rate, n_iterations)
    print("Linear Regression Parameters (theta):", theta)
    
    # Linear Regression Loss Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(range(n_iterations), loss_history)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Linear Regression Loss over Iterations')
    explanation = (
        "This plot shows the loss over iterations for the linear regression model. "
        "We use gradient descent to minimize the Mean Squared Error (MSE) loss. "
        "As iterations increase, the loss decreases, indicating that the model is "
        "learning the optimal parameters."
    )
    add_explanation_below(ax, explanation)
    plt.show()

    # Polynomial Regression
    degree = 3
    theta_poly, poly_features = polynomial_regression(X, y, degree)
    print(f"Polynomial Regression (Degree {degree}) Parameters:", theta_poly)

    # Polynomial Regression Fit Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    X_poly_plot = np.linspace(0, 2, 100).reshape(100, 1)
    X_poly_plot_b = poly_features.fit_transform(X_poly_plot)
    y_poly_plot = X_poly_plot_b.dot(theta_poly)
    ax.scatter(X, y, color='blue')
    ax.plot(X_poly_plot, y_poly_plot, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Polynomial Regression Fit')
    explanation = (
        "This plot shows the polynomial regression model fit with degree 3. "
        "The model transforms the input features into polynomial features and "
        "finds the best curve that fits the data."
    )
    add_explanation_below(ax, explanation)
    plt.show()

    # Ridge Regression
    alpha = 1.0
    ridge_model = ridge_regression(X, y, alpha)
    print("Ridge Regression Coefficients:", ridge_model.coef_)
    
    # Ridge Regression Fit Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X, y, color='blue')
    ax.plot(X, ridge_model.predict(X), color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Ridge Regression Fit')
    explanation = (
        "This plot shows the ridge regression model fit. Ridge regression is similar "
        "to linear regression but adds regularisation to prevent overfitting."
    )
    add_explanation_below(ax, explanation)
    plt.show()

    # Logistic Regression
    X_log, y_log = generate_logistic_data()
    logistic_model = logistic_regression(X_log, y_log)
    print("Logistic Regression Coefficients:", logistic_model.coef_)
    
    # Logistic Regression Fit Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X_log, y_log, color='blue')
    ax.plot(X_log, logistic_model.predict_proba(X_log)[:, 1], color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Probability')
    ax.set_title('Logistic Regression Fit')
    explanation = (
        "This plot shows the logistic regression model fit. The model predicts the "
        "probability of the data points belonging to a particular class."
    )
    add_explanation_below(ax, explanation)
    plt.show()

if __name__ == "__main__":
    train_models()

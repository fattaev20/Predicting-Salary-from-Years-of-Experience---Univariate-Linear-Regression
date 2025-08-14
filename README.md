# Predicting-Salary-from-Years-of-Experience---Univariate-Linear-Regression
This project predicts salaries based on **Years of Experience** using a **linear regression model trained with gradient descent**. 
It demonstrates the full process — from loading and visualizing data to implementing the gradient descent algorithm from scratch in Python.

## 📊 Dataset

The dataset (`Salary_dataset.csv`) contains:
- **YearsExperience**: Number of years a person has worked.
- **Salary**: Annual salary in USD.

Example:

| YearsExperience | Salary   |
|-----------------|----------|
| 1.1             | 39343.00 |
| 1.3             | 46205.00 |
| 1.5             | 37731.00 |


## 📐 Gradient Descent Formula

The model tries to minimize the cost function:

$$
J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (m x_i + b))^2
$$

Parameters are updated in each iteration:

$$
m := m - \alpha \cdot \frac{\partial J}{\partial m}
$$

$$
b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

Where:
- \( \alpha \) = learning rate  
- \( m \) = slope  
- \( b \) = intercept

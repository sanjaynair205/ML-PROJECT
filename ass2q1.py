import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# giving the dataset
x = np.array([12,23,34,45,56,67,78,89,123,134]).reshape(-1, 1)
y = np.array([240,1135,2568,4521,7865,9236,11932,14589,19856,23145])

#  Linear Regression
linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred_linear = linear_model.predict(x)

#  Polynomial Regression (Degree 2)
poly2 = PolynomialFeatures(degree=2)
x_poly2 = poly2.fit_transform(x)
poly2_model = LinearRegression()
poly2_model.fit(x_poly2, y)
y_pred_poly2 = poly2_model.predict(x_poly2)

# Polynomial Regression (Degree 3)
poly3 = PolynomialFeatures(degree=3)
x_poly3 = poly3.fit_transform(x)
poly3_model = LinearRegression()
poly3_model.fit(x_poly3, y)
y_pred_poly3 = poly3_model.predict(x_poly3)

#  SSE and R2 for each model
def calculate_metrics(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = r2_score(y_true, y_pred)
    return sse, r2

sse_linear, r2_linear = calculate_metrics(y, y_pred_linear)
sse_poly2, r2_poly2 = calculate_metrics(y, y_pred_poly2)
sse_poly3, r2_poly3 = calculate_metrics(y, y_pred_poly3)

#  results
print("Linear Regression:")
print(f"SSE: {sse_linear:.4f}, R2: {r2_linear:.4f}")

print("\nPolynomial Regression (Degree 2):")
print(f"SSE: {sse_poly2:.4f}, R2: {r2_poly2:.4f}")

print("\nPolynomial Regression (Degree 3):")
print(f"SSE: {sse_poly3:.4f}, R2: {r2_poly3:.4f}")

plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred_linear, color='red', label='Linear Fit')
plt.plot(x, y_pred_poly2, color='green', label='Polynomial Degree 2 Fit')
plt.plot(x, y_pred_poly3, color='orange', label='Polynomial Degree 3 Fit')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regression Models Comparison")
plt.legend()
plt.show()

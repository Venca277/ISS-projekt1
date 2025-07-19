import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data from the table
data = {
    "Koncentrace_hnojiva": [3,	4,	4,	5,	5,	5,	6,	6,	7,	8],
    "Teplota": [25, 27, 27, 29, 30, 25, 26, 26, 26, 27],
    "Vyska_rostliny": [1.35,	1.43,	1.55,	1.64,	1.67,	1.67,	1.79,	1.59,	1.77,	1.7]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Independent variables (X) and dependent variable (y)
X = df[["Koncentrace_hnojiva", "Teplota"]]
y = df["Vyska_rostliny"]

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# odhady parametrů roviny
beta_0 = model.intercept_  # Intercept
beta_1, beta_2 = model.coef_  # Coefficients for hnojiva and teplota

# rovnici nalezené roviny
regression_equation = f"{beta_0:.3f} + {beta_1:.3f}*x + {beta_2:.3f}*z"

# předpověď pro výšku rostliny při koncentraci hnojiva = 5 ml/l a teploty = 26 °C
predict_data = pd.DataFrame({"Koncentrace_hnojiva": [5], "Teplota": [26]})
predicted_value = model.predict(predict_data)[0]

# odhad rozptylu (sigma^2)
y_pred = model.predict(X)
residuals = y - y_pred
sigma_squared = np.var(residuals, ddof=2)  # Variance with degrees of freedom = 2

# koeficient procent variability výšky rostliny (R^2)
r_squared = r2_score(y, y_pred)

print("Regresní koeficienty:")
print(f"\u03B2_0 = {beta_0}")
print(f"\u03B2_1 = {beta_1}")
print(f"\u03B2_2 = {beta_2}")
print("\nRovnice regresní roviny:")
print(f"y = {regression_equation}")
print("\nOčekávaná výška rostliny:")
print(f"{predicted_value} m")
print(f"    > {predicted_value:.3f}")
print("\nOdhad rozptylu:")
print(f"\u03C3^2 = {sigma_squared}")
print(f"    > {sigma_squared:.3f}")
print("\nKoeficient determinace:")
print(f"R^2 = {r_squared}")
print(f"    > {r_squared:.3f}")
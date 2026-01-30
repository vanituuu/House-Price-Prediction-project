# ==========================================
#  House Price Prediction - ML Project
# Description: Predicts house prices using multiple regression models
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ==========================================
#  Load Dataset
# ==========================================
df = pd.read_csv("house_data_large.csv")

print("Shape of data:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nStatistical Summary:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())


# ==========================================
#  Exploratory Data Analysis (EDA)
# ==========================================

# Distribution of House Prices
plt.figure()
df["Price"].hist(bins=30)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Price vs Square Footage
plt.figure()
plt.scatter(df["Sqft"], df["Price"])
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Price vs Sqft")
plt.show()

# Bedrooms vs Price
plt.figure()
sns.boxplot(x=df["Bedrooms"], y=df["Price"])
plt.title("Bedrooms vs Price")
plt.show()

# Location vs Price
plt.figure()
sns.boxplot(x=df["LocationCode"], y=df["Price"])
plt.title("Location vs Price")
plt.show()

# Outlier Detection in Price
plt.figure()
sns.boxplot(df["Price"])
plt.title("Price Outliers")
plt.show()

# Feature Correlation with Target
corr = df.corr()["Price"].sort_values(ascending=False)
print("\nCorrelation with Price:\n", corr)


# ==========================================
#  Feature Selection
# ==========================================
X = df[["Sqft", "Bedrooms", "Bathrooms", "YearBuilt", "LocationCode"]]
y = df["Price"]


# ==========================================
#  Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# Feature Scaling
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# user input (Sqft, Bedrooms, Bathrooms, YearBuilt, LocationCode)
user_input = np.array([[2774, 4, 1, 2005, 3]])
user_input_scaled = scaler.transform(user_input)


# ==========================================
#  Model Training & Evaluation
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

for name, model in models.items():

    # Use scaled data only for Linear Regression and KNN
    if name in ["Linear Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        user_pred = model.predict(user_input_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        user_pred = model.predict(user_input)

    print("\n==============================")
    print(f" Model: {name}")
    print(f" Predicted Price for User Input: {user_pred[0]:,.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f" RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f" MAE: {mean_absolute_error(y_test, y_pred):.2f}")


# ==========================================
#  Feature Importance (Random Forest)
# ==========================================
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_names = X.columns

plt.figure()
plt.barh(feature_names, importances)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.show()
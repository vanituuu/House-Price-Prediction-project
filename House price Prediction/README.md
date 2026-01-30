🏠 House Price Prediction
Project Overview

This project predicts house prices based on features like square footage, number of bedrooms and bathrooms, year built, and location. The goal is to provide data-driven price predictions for buyers, sellers, and real estate platforms.

📊 Dataset

The dataset house_data_large.csv contains historical house data.

Key features:

Feature	Description
Sqft	Total square footage of the house
Bedrooms	Number of bedrooms
Bathrooms	Number of bathrooms
YearBuilt	Year the house was built
LocationCode	Encoded location/category of the house
Price	Target variable - house price
🔍 Exploratory Data Analysis (EDA)

Checked data shape, types, missing values, and statistics

Visualized distribution of house prices

Explored relationships:

Price vs Square Footage (scatter plot)

Price vs Bedrooms (box plot)

Price vs Location (box plot)

Checked for outliers using box plots

Checked correlation between features and target

🛠 Models Used

Five regression models were trained and evaluated:

Model	Notes
Linear Regression	Coefficient-based, scaled features used
Decision Tree	Handles non-linear patterns
Random Forest	Ensemble of trees, captures complex patterns
Gradient Boosting	Boosting ensemble for better accuracy
K-Nearest Neighbors	Distance-based, scaled features used
⚙️ Data Preprocessing

Split dataset into train (80%) and test (20%)

Standard Scaling applied to features for Linear Regression and KNN

Outliers were visually inspected but not removed for simplicity

📈 Model Evaluation Metrics

R² Score: Measures how well the model predicts variance in house prices

RMSE (Root Mean Squared Error): Measures prediction error magnitude

MAE (Mean Absolute Error): Average absolute error of predictions

🌟 Feature Importance

Feature importance calculated using Random Forest

Highlights which features contribute most to predicting house prices

Example insights:

Square Footage (Sqft) is the most important predictor

Location also significantly affects price

🏠 Prediction

Users can input a house’s features to get a predicted price

Example:

user_input = np.array([[2774, 4, 1, 2005, 3]])
predicted_price = model.predict(user_input)

📌 Key Learnings

Explored EDA and visualizations to understand data

Learned importance of scaling for some algorithms

Compared multiple regression models to find best performance

Understood feature importance and how features influence house price

Practiced model evaluation and prediction on real-world data

🔧 Tools & Libraries

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

💡 Use Cases

Home Buyers: Estimate fair house price

Real Estate Websites: Suggest prices for listings

Sellers: Set competitive prices

Banks: Evaluate property value for loans

Investors: Identify undervalued properties

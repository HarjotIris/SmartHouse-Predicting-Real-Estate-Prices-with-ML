import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

train_path = r"C:\Desktop\PRACTICE\house-prices-advanced-regression-techniques\train.csv"

test_path = r"C:\Desktop\PRACTICE\house-prices-advanced-regression-techniques\test.csv"

submission_path = r"C:\Desktop\PRACTICE\house-prices-advanced-regression-techniques\sample_submission.csv"

# Read the CSV files

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)

# print(train_df.info(), test_df.info(), train_df.head(), test_df.head())
# train.csv has 1460 rows & 81 columns (including the target variable SalePrice).
# test.csv has 1459 rows & 80 columns (without SalePrice).

# analysing the missing values
missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
missing_test = test_df.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)

# print(missing_test, missing_train)
# Missing Values Summary:
# -> Some columns (PoolQC, MiscFeature, Alley, Fence, FireplaceQu) have too many missing values and are likely not useful.
# -> Important columns like LotFrontage, GarageYrBlt, MasVnrArea, and BsmtQual have some missing values, so we must fill them.


# Drop columns with too many missing values
drop_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# fill missing numerical values with the median
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.to_list()
num_cols.remove("SalePrice")
for col in num_cols:
    train_df.fillna({col:train_df[col].median()}, inplace=True)
    test_df.fillna({col:test_df[col].median()}, inplace=True)

# fill missing categorical values with None
cat_cols = train_df.select_dtypes(include=["object"]).columns.to_list()
for col in cat_cols:
    train_df.fillna({col:"None"}, inplace=True)
    test_df.fillna({col:"None"}, inplace=True)

# Verify if there are still missing values
# missing_after_cleaning_train = train_df.isnull().sum().sum()
# missing_after_cleaning_test = test_df.isnull().sum().sum()

# print(missing_after_cleaning_train, missing_after_cleaning_test)


# One-Hot Encoding
train_encoded = pd.get_dummies(train_df, drop_first=True)
test_encoded = pd.get_dummies(test_df, drop_first=True)

# Ensure test_encoded has the same columns as train_encoded
test_encoded = test_encoded.reindex(columns=train_encoded.columns.drop("SalePrice"), fill_value=0)
# Verify shape after encoding
# print(train_encoded.shape, test_encoded.shape)
# Result:
# ((1460, 244), (1459, 244))
# Feature encoding is complete! Now both datasets have 244 numerical features after converting categorical variables.


# Split dataset
X = train_encoded.drop(columns=['SalePrice'])
y = train_encoded['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Ridge Regression
ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train, y_train)
ridge_mae = mean_absolute_error(y_val, ridge_model.predict(X_val))

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_mae = mean_absolute_error(y_val, rf_model.predict(X_val))

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_mae = mean_absolute_error(y_val, xgb_model.predict(X_val))

# Plot performance comparison
plt.figure(figsize=(8, 5))
plt.bar(["Ridge Regression", "Random Forest", "XGBoost"], [ridge_mae, rf_mae, xgb_mae], color=["blue", "green", "red"])
plt.ylabel("Mean Absolute Error (Lower is Better)")
plt.title("Model Performance Comparison")
plt.show()

# Predict using the best model (XGBoost)
test_predictions = np.maximum(xgb_model.predict(test_encoded), 0)
submission_df["SalePrice"] = test_predictions
submission_df.to_csv("house_price_predictions.csv", index=False)

print("Model Training Complete! Check 'house_price_predictions.csv' for results.")
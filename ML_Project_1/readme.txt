# Regression Model Selection and Training

This Python script demonstrates the process of selecting and training regression models using various algorithms to predict the target variable `MEDV` (median value of owner-occupied homes) based on different features.

## Overview

The script performs the following tasks:

1. **Data Loading and Preprocessing**:
   - Loads the dataset from a CSV file (`data.csv`).
   - Handles missing values by replacing them with the median value of the `RM` column.
   - Splits the data into training and testing sets.

2. **Model Selection and Training**:
   - **Model 1: Linear Regression**
   - **Model 2: Polynomial Regression**
     - Tests polynomial regression with degrees ranging from 2 to 5 and selects the best-performing degree based on RMSE.
   - **Model 3: Decision Tree Regression**
   - **Model 4: Random Forest Regression**
     - Random forest regression is identified as the best-performing model.

3. **Model Evaluation**:
   - Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for each model using the testing set.

4. **Final Model Training and Saving**:
   - Trains the selected model (Random Forest Regressor) on the entire dataset.
   - Saves the final model using joblib.

## Usage

1. Ensure that the `data.csv` file is present in the same directory as the script.
2. Run the script.
3. The script will print the MAE and RMSE for each model.
4. The final trained model will be saved as `Final_Model.joblib`.

## Files

- **data.csv**: Input dataset containing features and target variable.
- **Final_Model.joblib**: Saved final trained regression model.

## Dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `seaborn`
- `scikit-learn`
- `joblib`

## Author

ROHAN MOHAPATRA

House Price Regression Project

This project trains a Linear Regression model to predict house prices
based on various numerical features.

Features Used

-   Square_Footage
-   Num_Bedrooms
-   Num_Bathrooms
-   Year_Built
-   Lot_Size
-   Garage_Size
-   Neighborhood_Quality

What the Script Does

1.  Loads the dataset house_price_regression_dataset.csv
2.  Prints the first few rows
3.  Cleans column names (removes spaces)
4.  Prints dataset info and column list
5.  Removes duplicate rows
6.  Selects features and target variable (House_Price)
7.  Splits data into training and test sets (80/20 split)
8.  Trains a Linear Regression model
9.  Makes predictions on the test set
10. Saves the cleaned dataset as house_price_clean.csv

Requirements

Install required packages:

pip install pandas scikit-learn

How to Run

Run the script using:

python your_script_name.py

This will: - Display dataset details - Train the model - Print
predictions - Save the cleaned CSV file

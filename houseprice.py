import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1️⃣ Load CSV file
df = pd.read_csv('house_price_regression_dataset.csv')  
print("Original Data:")
print(df.head(), "\n")

# 🧹 Fix column names (strip spaces)
df.columns = df.columns.str.strip()

# 2️⃣ Inspect dataset
print("Columns in dataset:")
print(df.columns, "\n")

print("Data Info:")
print(df.info(), "\n")

# 3️⃣ Remove duplicates
df = df.drop_duplicates()

# 4️⃣ Define features and target
features = ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built',
            'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
target = 'House_Price'

# ✅ Now these columns will match correctly
X = df[features]
y = df[target]

# 5️⃣ Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 6️⃣ Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7️⃣ Make predictions on the test set
predictions = model.predict(X_test)
print("First 10 Predictions on Test Set:")
print(predictions[:10], "\n")

# 8️⃣ Save cleaned dataset
df.to_csv('house_price_clean.csv', index=False)
print("Cleaned dataset saved as 'house_price_clean.csv'")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
customers_df = pd.read_csv('customers.csv')
transactions_df = pd.read_csv('transactions.csv')

# --- Feature Engineering ---
print("Performing feature engineering...")

# Merge dataframes to get customer info for each transaction
df = pd.merge(transactions_df, customers_df, on='customer_id')

# For this model, we'll one-hot encode the products a customer ALREADY has.
# This creates a customer profile based on their current holdings.
customer_portfolio = pd.crosstab(df['customer_id'], df['product_name'])

# Merge portfolio back with customer details
full_df = pd.merge(customers_df, customer_portfolio, on='customer_id', how='left').fillna(0)

# --- Create the Training Set ---
# This is the crucial step. We are creating a dataset to predict the "next" product.
print("Creating the training set...")
potential_next_products = []

all_products = transactions_df['product_name'].unique()

for _, row in full_df.iterrows():
    customer_id = row['customer_id']
    
    # --- FIX STARTS HERE ---
    # Check if customer has any products before trying to access the portfolio
    if customer_id in customer_portfolio.index:
        # Get products for customers with a transaction history
        owned_products = customer_portfolio.loc[customer_id][customer_portfolio.loc[customer_id] > 0].index.tolist()
    else:
        # If customer has no transactions, their portfolio is empty
        owned_products = []
    # --- FIX ENDS HERE ---
    
    # Identify products the customer does NOT own
    products_to_predict = [p for p in all_products if p not in owned_products]
    
    for product in products_to_predict:
        potential_next_products.append({
            'customer_id': customer_id,
            'potential_product': product
        })

# Create a dataframe of potential next products
predict_df = pd.DataFrame(potential_next_products)

# Merge with customer features
training_df = pd.merge(full_df, predict_df, on='customer_id')

# The target variable is 1 if the 'potential_product' is in the customer's portfolio,
# but since we filtered for products they DON'T own, the target is implicitly 0 for all these rows.
# To create a real target, we must find which of these "potential" products were LATER acquired.
# For simplicity in this project, we'll assume the LATEST product acquired is the target.

# Find the last product acquired by each customer
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
last_transactions = df.loc[df.groupby('customer_id')['transaction_date'].idxmax()]

# Create the target variable
target_map = last_transactions[['customer_id', 'product_name']].rename(columns={'product_name': 'target_product'})
training_df = pd.merge(training_df, target_map, on='customer_id', how='left')

# Target is 1 if potential_product matches the target_product
training_df['target'] = (training_df['potential_product'] == training_df['target_product']).astype(int)

# Create a dataframe of potential next products
predict_df = pd.DataFrame(potential_next_products)

# Merge with customer features
training_df = pd.merge(full_df, predict_df, on='customer_id')

# The target variable is 1 if the 'potential_product' is in the customer's portfolio,
# but since we filtered for products they DON'T own, the target is implicitly 0 for all these rows.
# To create a real target, we must find which of these "potential" products were LATER acquired.
# For simplicity in this project, we'll assume the LATEST product acquired is the target.

# Find the last product acquired by each customer
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
last_transactions = df.loc[df.groupby('customer_id')['transaction_date'].idxmax()]

# Create the target variable
target_map = last_transactions[['customer_id', 'product_name']].rename(columns={'product_name': 'target_product'})
training_df = pd.merge(training_df, target_map, on='customer_id', how='left')

# Target is 1 if potential_product matches the target_product
training_df['target'] = (training_df['potential_product'] == training_df['target_product']).astype(int)

# --- Data Preparation for XGBoost ---
print("Preparing data for the model...")

# Encode the categorical 'potential_product' feature
le = LabelEncoder()
training_df['potential_product_encoded'] = le.fit_transform(training_df['potential_product'])

# Define features (X) and target (y)
features = [col for col in full_df.columns if col not in ['customer_id', 'name', 'registration_date']]
features.append('potential_product_encoded')

X = training_df[features]
y = training_df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- Train the XGBoost Model ---
print("Training XGBoost model...")

# FIX: Calculate scale_pos_weight to handle extreme class imbalance
# This tells the model how many times more to weight the minority class (target=1)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight  # Add this parameter to handle imbalance
)

model.fit(X_train, y_train)
# --- Evaluate Model ---
accuracy = model.score(X_test, y_test)
print(f"\nModel training complete. Accuracy: {accuracy:.4f}")


# --- Save Model and Artifacts ---
print("Saving model and required artifacts...")

# Save the trained model
model.save_model("xgb_model.json")

# Save the label encoder
joblib.dump(le, 'product_label_encoder.pkl')

# Save the feature columns
joblib.dump(list(X.columns), 'model_columns.pkl')

print("\nArtifacts saved successfully: xgb_model.json, product_label_encoder.pkl, model_columns.pkl")
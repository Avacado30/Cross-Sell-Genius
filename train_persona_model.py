import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

print("Loading data...")
customers_df = pd.read_csv('customers.csv')
transactions_df = pd.read_csv('transactions.csv')

# --- Feature Engineering ---
print("Engineering features for persona modeling...")
# Calculate number of products for each customer
product_counts = transactions_df.groupby('customer_id')['product_name'].count().reset_index()
product_counts.rename(columns={'product_name': 'product_count'}, inplace=True)

# Merge features
df = pd.merge(customers_df, product_counts, on='customer_id', how='left').fillna(0)

# Select features for clustering
features_for_clustering = ['age', 'income', 'tenure_days', 'product_count']
X = df[features_for_clustering]

# --- Scale the Data ---
# Scaling is crucial for distance-based algorithms like K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train K-Means Model ---
# We'll create 4 personas for this example
N_CLUSTERS = 4
print(f"Training K-Means model with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Assign the persona label to each customer
df['persona'] = kmeans.labels_

print("\nPersona Analysis Complete. Sample of customers with their persona:")
print(df[['customer_id', 'age', 'income', 'product_count', 'persona']].head())

# --- Save the Model and Scaler ---
print("\nSaving persona model and scaler...")
joblib.dump(kmeans, 'persona_model.pkl')
joblib.dump(scaler, 'persona_scaler.pkl')
# Save the customer-persona mapping for easy lookup in the API
df[['customer_id', 'persona']].to_csv('customer_personas.csv', index=False)

print("Persona model artifacts saved successfully.")
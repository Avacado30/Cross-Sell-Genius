import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Saved Artifacts ---
print("Loading saved model and artifacts...")
# Load the trained model
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

# Load the label encoder for products
product_encoder = joblib.load('product_label_encoder.pkl')

# Load the list of columns the model was trained on
model_columns = joblib.load('model_columns.pkl')
print("Artifacts loaded successfully.")

# --- 3. Load Data for Feature Generation ---
# In a real application, this would query a live database.
# In the data loading section, add:
print("Loading persona data...")
personas_df = pd.read_csv('customer_personas.csv').set_index('customer_id')

# Define persona descriptions (you can make these more creative)
PERSONA_MAP = {
    0: "Young Professional Accumulator",
    1: "Established Family Planner",
    2: "Cautious Newcomer",
    3: "Seasoned High-Value Customer"
}

# Add this new endpoint function
@app.route('/persona/<int:customer_id>', methods=['GET'])
def get_persona(customer_id):
    """API endpoint to get the financial persona for a given customer."""
    try:
        if customer_id not in personas_df.index:
            return jsonify({"error": "Customer ID not found"}), 404
        
        persona_label = personas_df.loc[customer_id, 'persona']
        persona_name = PERSONA_MAP.get(int(persona_label), "Unknown Persona")

        print(f"Persona request for {customer_id}: {persona_name}")
        return jsonify({
            "customer_id": customer_id,
            "persona": persona_name
        })
    except Exception as e:
        return jsonify({"error": "An error occurred.", "details": str(e)}), 500

# For our project, we load the CSVs into memory.
print("Loading data for feature lookups...")
customers_df = pd.read_csv('customers.csv')
transactions_df = pd.read_csv('transactions.csv')
# Create the customer portfolio for feature lookups
customer_portfolio = pd.crosstab(transactions_df['customer_id'], transactions_df['product_name'])
full_features_df = pd.merge(customers_df, customer_portfolio, on='customer_id', how='left').fillna(0)
print("Data loaded.")

# --- 4. Create the Prediction Endpoint ---
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to recommend the next best product for a given customer.
    Expects a JSON payload with a 'customer_id'.
    e.g., {"customer_id": 1001}
    """
    try:
        # Get customer_id from the request
        data = request.get_json()
        customer_id = int(data['customer_id'])
        print(f"\nReceived recommendation request for customer_id: {customer_id}")

        # --- Recreate Features for the Specific Customer ---
        # Find the customer's base features
        if customer_id not in full_features_df['customer_id'].values:
            return jsonify({"error": "Customer ID not found"}), 404

        customer_features = full_features_df[full_features_df['customer_id'] == customer_id]

        # Identify products the customer does NOT own
        owned_products = customer_portfolio.loc[customer_id][customer_portfolio.loc[customer_id] > 0].index.tolist()
        all_products = transactions_df['product_name'].unique()
        products_to_predict = [p for p in all_products if p not in owned_products]

        if not products_to_predict:
            return jsonify({
                "customer_id": customer_id,
                "recommendation": "Customer already owns all available products."
            })

        # --- Prepare Data for Prediction ---
        # Create a dataframe with a row for each potential product
        prediction_input_list = []
        for product in products_to_predict:
            # Create a copy of the customer's features
            features_row = customer_features.to_dict('records')[0]
            # Add the potential product to the features
            features_row['potential_product'] = product
            prediction_input_list.append(features_row)

        predict_df = pd.DataFrame(prediction_input_list)

        # One-hot encode the potential product names
        predict_df['potential_product_encoded'] = product_encoder.transform(predict_df['potential_product'])
        
        # Ensure columns match the training format
        predict_df_aligned = predict_df[model_columns]

        # --- Make Prediction ---
        # Predict the probability of acquiring each potential product
        probabilities = model.predict_proba(predict_df_aligned)[:, 1] # Get probability of the '1' class

        # Find the product with the highest probability
        best_product_index = np.argmax(probabilities)
        best_product_name = products_to_predict[best_product_index]
        confidence_score = probabilities[best_product_index]
        
        print(f"Prediction complete. Best product: {best_product_name}, Score: {confidence_score:.4f}")

        # --- Return the Result ---
        return jsonify({
            "customer_id": customer_id,
            "recommendation": best_product_name,
            "confidence_score": float(confidence_score) # Convert numpy float to native Python float
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500


# --- 5. Run the Flask App ---
if __name__ == '__main__':
    # Use port 5001 to avoid conflicts with other common services
    app.run(debug=True, port=5001)
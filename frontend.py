import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Cross-Sell Genius",
    page_icon="🧠",
    layout="centered"
)

# --- UI Elements ---
st.title("🧠 Cross-Sell Genius")
st.markdown("A proactive customer intelligence suite for InCred.")

st.subheader("Existing Customer Recommendation")
st.markdown("Enter a customer ID to get the next best product recommendation and persona analysis.")


# The URL of your backend API
API_URL = "http://127.0.0.1:5001/recommend"

# Input box for customer ID
customer_id_input = st.text_input(
    label="Customer ID",
    placeholder="e.g., 1001"
)

# Recommendation button
if st.button("Get Recommendation"):
    # --- Input Validation ---
    if not customer_id_input.strip():
        st.error("Please enter a Customer ID.")
    elif not customer_id_input.isdigit():
        st.error("Customer ID must be a number.")
    else:
        # --- API Call ---
        with st.spinner('Thinking...'):
            try:
                # Prepare the data payload for the POST request
                payload = {"customer_id": int(customer_id_input)}

                # Send request to the Flask API
                response = requests.post(API_URL, json=payload)

                # --- Handle Response ---
                if response.status_code == 200:
                    recommendation = response.json()
                    product = recommendation.get("recommendation")
                    score = recommendation.get("confidence_score", 0)

                    # --- 1. Display the product recommendation ---
                    st.success(f"**Recommended Product: {product}**")
                    st.info(f"Confidence Score: {score:.2%}")

                    # --- 2. Fetch and Display Persona ---
                    try:
                        # Use the same customer ID from the input box
                        customer_id = int(customer_id_input)
                        # Make a new request to the persona endpoint
                        persona_response = requests.get(f"http://127.0.0.1:5001/persona/{customer_id}")

                        if persona_response.status_code == 200:
                            persona_name = persona_response.json().get("persona")
                            st.subheader("Inferred Customer Persona")
                            st.metric(label="Persona", value=persona_name)
                    except Exception as e:
                        # If the persona service fails, don't crash the app.
                        st.warning("Could not retrieve customer persona.")
                        print(f"Persona fetch error: {e}") # For debugging

                elif response.status_code == 404:
                    st.error("Customer ID not found. Please try another ID.")

                else:
                    # Handle other potential server errors
                    st.error(f"An error occurred on the server. Status code: {response.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the recommendation service. Please ensure the backend is running. Details: {e}")


# --- Onboarding Section for New Customers ---
st.divider()

st.subheader("New Customer? Get Started Here")
st.markdown("Enter your details to find the best first product for you.")

col1, col2 = st.columns(2)
with col1:
    age_input = st.number_input("Your Age", min_value=18, max_value=100, value=25)
with col2:
    income_input = st.number_input("Your Annual Income (INR)", min_value=50000, value=500000, step=10000)

if st.button("Find My First Product"):
    with st.spinner('Analyzing profile...'):
        try:
            # New API URL for onboarding
            ONBOARD_API_URL = "http://127.0.0.1:5001/onboard"
            payload = {"age": age_input, "income": income_input}
            response = requests.post(ONBOARD_API_URL, json=payload)

            if response.status_code == 200:
                recommendation = response.json().get("recommendation")
                st.success(f"**Recommended First Product: {recommendation}**")
            else:
                st.error("Could not get a recommendation at this time.")
        except Exception as e:
            st.error("Failed to connect to the service.")
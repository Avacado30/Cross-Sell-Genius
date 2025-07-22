import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker('en_IN')

# --- Configuration ---
NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 3000
PRODUCTS = [
    "Savings Account", "Credit Card", "Personal Loan", "Home Loan",
    "Education Loan", "Fixed Deposit", "Mutual Funds", "Stock Trading Account"
]

# --- Generate Customers Data ---
print("Generating customer data...")
customers = []
for i in range(NUM_CUSTOMERS):
    registration_date = fake.date_time_between(start_date='-5y', end_date='now')
    tenure_days = (datetime.now() - registration_date).days
    customers.append({
        'customer_id': 1001 + i,
        'name': fake.name(),
        'age': random.randint(22, 65),
        'income': round(random.uniform(300000, 2500000) / 10000) * 10000, # Annual income
        'registration_date': registration_date,
        'tenure_days': tenure_days
    })

customers_df = pd.DataFrame(customers)

# --- Generate Transactions Data ---
print("Generating transaction data...")
transactions = []
for _ in range(NUM_TRANSACTIONS):
    customer = random.choice(customers)
    # Older customers are more likely to have more products
    num_products = random.randint(1, 2) if customer['tenure_days'] < 365 else random.randint(1, 5)

    # Higher income customers more likely to have investment products
    product_pool = PRODUCTS
    if customer['income'] > 1500000:
        product_pool += ["Mutual Funds", "Stock Trading Account"] * 2 # Increase probability

    owned_products = random.sample(product_pool, k=min(num_products, len(product_pool)))

    for product in owned_products:
        transactions.append({
            'transaction_id': 20001 + len(transactions),
            'customer_id': customer['customer_id'],
            'product_name': product,
            'transaction_date': fake.date_time_between(start_date=customer['registration_date'], end_date='now')
        })

# Remove duplicate customer-product pairs, keeping the first transaction
transactions_df = pd.DataFrame(transactions)
transactions_df = transactions_df.sort_values('transaction_date').drop_duplicates(subset=['customer_id', 'product_name'], keep='first')


# --- Save to CSV ---
print("Saving data to CSV files...")
customers_df.to_csv('customers.csv', index=False)
transactions_df.to_csv('transactions.csv', index=False)

print("\nData generation complete!")
print(f"Generated {len(customers_df)} customers.")
print(f"Generated {len(transactions_df)} unique transactions.")

print("\nCustomers DataFrame Head:")
print(customers_df.head())

print("\nTransactions DataFrame Head:")
print(transactions_df.head())
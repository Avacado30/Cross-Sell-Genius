import pandas as pd
import mysql.connector
from mysql.connector import Error

def create_db_connection(host_name, user_name, user_password, db_name):
    """Creates a connection to the database."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_query(connection, query, data):
    """Executes a SQL query."""
    cursor = connection.cursor()
    try:
        cursor.executemany(query, data)
        connection.commit()
        print(f"{cursor.rowcount} rows inserted successfully.")
    except Error as e:
        print(f"The error '{e}' occurred")

# --- Main Script ---

# !! IMPORTANT: Replace with your MySQL credentials !!
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "Aashu@2004"
DB_NAME = "cross_sell_db"

# Establish connection
conn = create_db_connection(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

# Load data from CSV
customers_df = pd.read_csv('customers.csv')
transactions_df = pd.read_csv('transactions.csv')

# Convert DataFrames to list of tuples for insertion
customers_data = [tuple(row) for row in customers_df.to_numpy()]
transactions_data = [tuple(row) for row in transactions_df.to_numpy()]

# Define SQL insert statements
insert_customers = """
INSERT INTO customers (customer_id, name, age, income, registration_date, tenure_days)
VALUES (%s, %s, %s, %s, %s, %s);
"""

insert_transactions = """
INSERT INTO transactions (transaction_id, customer_id, product_name, transaction_date)
VALUES (%s, %s, %s, %s);
"""

# Execute queries
print("\nInserting customer data...")
execute_query(conn, insert_customers, customers_data)

print("\nInserting transaction data...")
execute_query(conn, insert_transactions, transactions_data)

# Close the connection
if conn.is_connected():
    conn.close()
    print("\nMySQL connection is closed.")
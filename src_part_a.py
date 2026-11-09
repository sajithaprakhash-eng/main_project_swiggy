import os
import sqlite3
import pandas as pd
import numpy as np

# Load raw tables
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

print("--- RAW AUDIT ---")
print(f"Orders shape: {orders.shape}")
print(f"Exact duplicate rows in orders: {orders.duplicated().sum()}")
print(f"Pending status count: {(orders['status'] == 'pending').sum()}")
print(f"Distinct order_ids: {orders['order_id'].nunique()}")
# Clean customers
customers['city'] = customers['city'].str.strip().str.title()

def clean_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g_lower = str(g).strip().lower()
    if g_lower in ['m', 'male']:
        return "male"
    elif g_lower in ['f', 'female']:
        return "female"
    return "unknown"

customers['gender'] = customers['gender'].apply(clean_gender)

# Age cleaning
customers['age'] = pd.to_numeric(customers['age'], errors='coerce')
customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = np.nan
city_medians = customers.groupby('city')['age'].transform('median')
customers['age'] = customers['age'].fillna(city_medians)
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
# Clean orders
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
orders = orders.drop_duplicates()

# Merge with products to get price_inr and category
orders = orders.merge(products[['product_id', 'price_inr', 'category']], on='product_id', how='left')

# Keep latest order_ts per order_id
orders = orders.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')

# Amount cleaning & imputation
orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
missing_amt = orders['order_amount'].isna()
calculated_amt = np.round(orders['price_inr'] * orders['quantity'] * (1 - orders['discount_pct'] / 100.0), 2)
orders.loc[missing_amt, 'order_amount'] = calculated_amt[missing_amt]

# Delivery minutes cleaning & imputation
orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan

# Completed category median imputation
cat_medians = orders[orders['status'] == 'completed'].groupby('category')['delivery_minutes'].transform('median')
comp_missing = (orders['status'] == 'completed') & (orders['delivery_minutes'].isna())
orders.loc[comp_missing, 'delivery_minutes'] = cat_medians[comp_missing]

# Select final columns
orders_clean = orders[['order_id', 'customer_id', 'product_id', 'quantity', 'order_ts', 'status', 'discount_pct', 'order_amount', 'delivery_minutes']].copy()
orders_clean['order_ts'] = orders_clean['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
os.makedirs("data/clean", exist_ok=True)
customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_clean.to_csv("data/clean/orders.csv", index=False)

db_path = "data/instamart.db"
conn = sqlite3.connect(db_path)
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
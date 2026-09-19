import os
import pandas as pd
import numpy as np

np.random.seed(42)
n_rows = 10000
timestamps = pd.date_range(start="2022-10-17 08:00:00", periods=n_rows, freq="15min")
banks = ["BankA", "BankB", "BankC", "BankD", "BankE"]
formats = ["ACH", "Wire", "Cash", "Credit Card", "Reinvestment"]

data = []
for i in range(n_rows):
    # Simulate some high-value wire transfers vs normal retail transactions
    is_suspicious_pattern = (i % 10 == 0)
    if is_suspicious_pattern:
        amount = np.random.uniform(9000.0, 9999.0)  # Near-threshold structuring
        payment_format = "Wire"
    else:
        amount = np.random.uniform(10.0, 1500.0)      # Normal spending
        payment_format = np.random.choice(formats)
        
    data.append({
        "Timestamp": timestamps[i],
        "From Bank": np.random.choice(banks),
        "Account": f"8000{i:03d}X",
        "To Bank": np.random.choice(banks),
        "Account_duplicated_0": f"8000{(i+5)%100:03d}Y",
        "Amount Received": round(amount, 2),
        "Receiving Currency": "USD",
        "Amount Paid": round(amount, 2),
        "Payment Currency": "USD",
        "Payment Format": payment_format
    })

os.makedirs("input", exist_ok=True)
df = pd.DataFrame(data)
df.to_csv("input/unlabeled_transactions.csv", index=False)
print("✅ input/unlabeled_transactions.csv generated without any laundering labels!")
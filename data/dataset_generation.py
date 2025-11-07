import pandas as pd
import random
from textblob import TextBlob
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Function to generate feedback and churn
def generate_row(i):
    gender = random.choice(["Male", "Female"])
    age = random.randint(18, 65)
    tenure = random.randint(1, 48)
    subscription = random.choice(["Basic", "Standard", "Premium"])
    monthly = random.choice([399, 499, 599, 699, 799, 899, 999])
    total = monthly * tenure
    last_days = random.randint(1, 30)
    
    feedback_options = [
        "Excellent service and quick response",
        "Poor quality of service and support",
        "Very satisfied with subscription",
        "Not happy with service delay",
        "Support team is great and helpful",
        "Average experience",
        "Frequent connectivity issues",
        "Good support but sometimes slow",
        "Very loyal customer",
        "Unresponsive customer support",
        "Quick issue resolution",
        "Unhappy with delay in response",
        "Connectivity issues frequently",
        "Good overall experience",
        "Not worth the price",
        "Happy with service"
    ]
    
    feedback = random.choice(feedback_options)
    churn = random.choice(["Yes", "No"])  # as per project description
    
    # Perform basic sentiment analysis using TextBlob
    sentiment_score = round(TextBlob(feedback).sentiment.polarity, 2)
    
    return [f"C{i:03}", gender, age, tenure, subscription, monthly, total, last_days, feedback, sentiment_score, churn]


# Generate initial 500 rows
data = [generate_row(i) for i in range(1, 501)]

# Generate additional 500 rows (IDs from 501 to 1000)
additional_data = [generate_row(i) for i in range(501, 1001)]

# Combine datasets
data.extend(additional_data)

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Customer_ID", "Gender", "Age", "Tenure", "Subscription_Type",
    "Monthly_Charges", "Total_Spend", "Last_Interaction_Days", 
    "Feedback", "Sentiment", "Churn"
])

# Save dataset
file_path = "data/customer_churn.csv"
df.to_csv(file_path, index=False)

print(f"✅ Dataset with 1000 rows (including Sentiment score) saved successfully: {file_path}")
print(df.head(10))
print(df.tail(10))

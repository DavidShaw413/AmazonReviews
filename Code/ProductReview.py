# STEP 1: Load and Prepare the Amazon Electronics Ratings Dataset
import pandas as pd
import matplotlib.pyplot as plt

# File path to the dataset
file_path = "Data/ratings_Electronics.csv"

# Manually assign proper column names since the file has no header
column_names = ['user_id', 'product_id', 'rating', 'timestamp']
df = pd.read_csv(file_path, names=column_names)

# Convert Unix timestamp to readable date
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Preview the shape and column names
print(f"Dataset loaded. {df.shape[0]} rows and {df.shape[1]} columns.\n")
print("First few rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Total number of reviews
print("\nTotal reviews:", len(df))

# Check unique rating values
print("Unique rating values:", df['rating'].unique())

# Frequency count of each rating
rating_counts = df['rating'].value_counts().sort_index()
print("\nRating counts:")
print(rating_counts)

# STEP 2: Visualize Rating Distribution
plt.figure(figsize=(8, 5))
rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# STEP 3A: Users who only leave 5-star reviews
only_5s = df.groupby('user_id')['rating'].apply(lambda x: (x == 5).all())
suspicious_users = only_5s[only_5s].index
print(f"\nUsers who only gave 5-star reviews: {len(suspicious_users)}")

# Optionally preview some of them
print("\nSample of users who only gave 5-star reviews:")
print(suspicious_users[:5])

# STEP 3B: Check for duplicate (user_id, product_id) pairs
duplicate_reviews = df.duplicated(subset=['user_id', 'product_id'])
num_duplicates = duplicate_reviews.sum()
print(f"\nDuplicate user/product review pairs found: {num_duplicates}")

# Show a few examples
if num_duplicates > 0:
    print("\nSample duplicate entries:")
    print(df[duplicate_reviews].head())

# STEP 4: Flag suspicious 5-star-only users
df['only_5_star_user'] = df['user_id'].isin(suspicious_users)

# STEP 5: Export suspicious users to CSV (unique list)
suspicious_df = pd.DataFrame(suspicious_users, columns=['user_id'])
suspicious_df.to_csv("Data/suspicious_5_star_users.csv", index=False)
print("\nSuspicious user list exported to: Data/suspicious_5_star_users.csv")

# STEP 6: QA Summary Report
total_reviews = len(df)
rating_counts = df['rating'].value_counts().sort_index()
five_star_percentage = (rating_counts[5] / total_reviews) * 100 if 5 in rating_counts else 0

print("\n===== QA SUMMARY REPORT =====")
print(f"Total reviews: {total_reviews}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"% of 5-star reviews: {five_star_percentage:.2f}%")
print(f"Users who only gave 5-star reviews: {len(suspicious_users)}")
print(f"Duplicate (user, product) entries: {num_duplicates}")
print("=============================\n")

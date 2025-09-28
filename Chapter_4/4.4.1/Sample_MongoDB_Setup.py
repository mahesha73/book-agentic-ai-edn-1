# Sample_MongoDB_Setup.py
# Section 4.4.1
# Page 109

from pymongo import MongoClient

# Connect to MongoDB server
client = MongoClient("mongodb://localhost:27017")

# Select database and collection
db = client["agentic_ai"]
learning_collection = db["learning_progress"]

# Ensure Indexes for Efficient Querying (Optional but Recommended)
learning_collection.create_index("user_id")
learning_collection.create_index("timestamp")

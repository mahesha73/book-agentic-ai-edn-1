# Sample_UpdatingMemory.py
# Section 4.4.4
# Page 111

from datetime import datetime

learning_data = {
    "user_id": user_id,
    "session_id": session_id,
    "progress_summary": summary_text,
    "timestamp": datetime.utcnow()
}

learning_collection.insert_one(learning_data)

# Testing_AI_Tutor.py
# Section 4.5.4
# Page 117-118

import requests

API_URL = "http://127.0.0.1:8000/query"
FEEDBACK_URL = "http://127.0.0.1:8000/feedback"

def ask_tutor(question):
    payload = {"user_input": question}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        print("Tutor response:\n", data.get("response"))
        print("\nLearning summary:\n", data.get("learning_summary"))
        return data.get("interaction_id")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with tutor API: {e}")
        return None

def send_feedback(interaction_id, rating, comments=None):
    payload = {
        "interaction_id": interaction_id,
        "rating": rating,
        "comments": comments
    }

    try:
        response = requests.post(FEEDBACK_URL, json=payload)
        response.raise_for_status()
        print("Feedback response:", response.json().get("message"))
    except requests.exceptions.RequestException as e:
        print(f"Error sending feedback: {e}")

if __name__ == "__main__":
    print("Personalized Learning Tutor Client")
    print("Type ‘exit’ to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        interaction_id = ask_tutor(user_input)
        if interaction_id:
            rating = input("Rate the response (1-5): ")
            comments = input("Any comments? (optional): ")
            send_feedback(interaction_id, int(rating), comments)

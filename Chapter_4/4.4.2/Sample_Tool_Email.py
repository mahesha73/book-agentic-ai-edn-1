# Sample_Tool_Email.py
# Section 4.4.2
# Page 110

from langchain.tools import Tool, smtplib

def send_email(to_address: str, subject: str, body: str) -> str:
    try:
        with smtplib.SMTP("smtp.example.com") as server:
            server.login("user", "password")
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail("from@example.com", to_address, message)
            return "Email sent successfully."
        except Exception as e:
            return f"Failed to send email: {str(e)}"

email_tool = Tool(
    name="SendEmail",
    func=send_email,
    description="Sends an email to the specifi ed address with subject and body."
)

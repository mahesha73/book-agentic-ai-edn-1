# Sample_Prompt_Template.py
# Section 4.3.1
# Page 101

from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["input", "chat_history"],
        template="""
            You are a helpful assistant.
            Use the conversation history and new user input to decide the next action.

            Conversation history: {chat_history}
            User input: {input}

            What will you do next?
        """
)

# Sample_FewShotPromptTemplate.py
# Section 4.3.1
# Page 100

from langchain.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["text", "label"],
    template="Text: {text}\nLabel: {label}\n"
    )

few_shot_prompt = FewShotPromptTemplate(
    examples=[
        {"text": "I love this product!", "label": "Positive"},
        {"text": "This is the worst experience.", "label": "Negative"},
        ],

    example_prompt=example_prompt,
    input_variables=["input_text"],
    prefix="Classify the sentiment of the following text.",
    suffix="Text: {input_text}\nLabel:",
    example_separator="\n"

)

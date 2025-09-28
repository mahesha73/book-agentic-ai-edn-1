# agentic_ai_app.py
# Section 4.5.1
# Page 113

# =========================
# IMPORT Necessary Packages
# =========================
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from pymongo import MongoClient
import smtplib
from email.mime.text import MIMEText

# =========================
# SENSE PHASE (Steps 1 to 9)
# =========================

# Step 1: Setup logging to output info and debug messages to console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 2: Load environment variables for configuration and secrets
# Path to PDF file with educational content
PDF_PATH = os.getenv("PDF_PATH", "educational_material.pdf")

# MongoDB connection string
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

# MongoDB database name
MONGODB_DB = os.getenv("MONGODB_DB", "agentic_ai_db")

# MongoDB collection name
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION",
 							"learning_progress")

# Email credentials (to send email)
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "your_email@example.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your_email_password")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 465))  #SSL

# OpenAI API key for LLM access
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 3: Validate that OpenAI API key is set, else raise error and stop execution
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Step 4: Load and chunk educational PDF material
logger.info(f"Loading educational material from {PDF_PATH}...")
# Initialize PDF loader with file path
loader = PyPDFLoader(PDF_PATH)
# Load all pages as separate documents
documents = loader.load()

# Step 5: Split documents into smaller chunks for better embedding and retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
 						chunk_overlap=100)
# Split documents into chunks of ~1000 chars with 100 overlap
chunks = text_splitter.split_documents(documents)
logger.info(f"Split educational material into {len(chunks)}
 				chunks.")

# Step 6: Create embeddings for chunks and store in Chroma vector store
logger.info("Creating embeddings and initializing Chroma vector
 					store...")
# Initialize OpenAI embeddings model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# Create vector store
vectorstore = Chroma.from_documents(chunks, embedding_model,
 					 collection_name="learning_material_docs")

# Step 7: Setup FastAPI app to receive user input
app = FastAPI(title="Personalized Learning Tutor AI")

# Step 8: Define request model for input validation
class QueryRequest(BaseModel):
    user_input: str  # User's input query or command

# Step 9: Load conversation memory (context sensing)
memory = ConversationBufferMemory(memory_key="chat_history")

# =========================
# PLAN PHASE (Steps 1 to 7)
# =========================

# Step 1: Initialize LLM with monitoring
llm = OpenAI(
    temperature=0,  # Deterministic responses
    openai_api_key=OPENAI_API_KEY,  # API key for authentication
    callbacks=[StdOutCallbackHandler()]  # Log LLM calls and tokens
 																to stdout for monitoring
)

# Step 2: Load summarization chain
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

# Step 3: Define tools
def query_vectorstore(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    combined_text = "\n\n".join([doc.page_content for doc in
 												results])
    return combined_text

def summarize_input(user_input: str) -> str:
    context = query_vectorstore(user_input)
    docs = [user_input, context]
    summary = summary_chain.run(docs)
    return summary

def save_to_mongodb(data: dict) -> str:
    collection.insert_one(data)
    return "Learning progress saved to MongoDB."

def email_summary(summary: str, to_email: str) -> str:
    send_email("Your Learning Progress Report", summary, to_email)
    return f"Learning progress emailed to {to_email}."

# Step 4: Define LangChain tools
tools = [
    Tool(
        name="SummarizeInput",
        func=summarize_input,
        description="Summarizes student input with educational
 							context"
    ),
    Tool(
        name="SaveToMongoDB",
        func=save_to_mongodb,
        description="Saves learning progress to MongoDB"
    ),
    Tool(
        name="EmailSummary",
        func=email_summary,
        description="Sends learning progress summary via email"
    )
]

# Step 5: Define prompt template for planning
prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
             You are a personalized learning tutor AI. Use the
             conversation history and new student input to decide
             what to do next.

 					Conversation history:
 					{chat_history}

 					Student input:
 					{input}

 					Decide the best action or tool to use and respond
 					accordingly.
 	  """
)

# Step 6: Create LLM chain with prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# =========================
# ACT PHASE (Steps 1 to 7)
# =========================

# Step 1: Setup MongoDB client (environment for acting)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]
collection = db[MONGODB_COLLECTION]
eval_collection = db["interaction_logs"]
feedback_collection = db["feedback"]

# Step 2: Define email sending function
def send_email(subject: str, body: str, to_email: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    with smtplib.SMTP_SSL(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as
        server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
                       logger.info(f"Email sent to {to_email}")

# Step 3: Initialize the agent with tools, LLM, memory, and agent type
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True
)

# Step 4: Define `/query` endpoint to receive input, plan, act, learn, and respond
@app.post("/query")
async def query_agent(request: QueryRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Input cannot be
                                                     empty")

    # Step 5: Handle special command to email progress report
    if user_input.lower().startswith("email progress to"):
        to_email = user_input.split("email progress to")[-1].strip()
        learning_summary =
                   memory.load_memory_variables({})["chat_history"]
                   email_response = email_summary(learning_summary,
                                                  to_email)
        return {"response": email_response}

    # Step 6: Plan next action using LLM chain
    plan = llm_chain.run(input=user_input,
      chat_history=memory.load_memory_variables({})["chat_history"])
      logger.info(f"Plan: {plan}")

    # Step 7: Act - run agent to generate response and use tools
    response = agent.run(user_input)

    # Learn: summarize learning progress so far
    full_history = memory.load_memory_variables({})["chat_history"]
    learning_summary = summary_chain.run([full_history])

    # Save learning progress to MongoDB
    save_to_mongodb({
        "student_input": user_input,
        "tutor_response": response,
        "learning_summary": learning_summary
    })

    # Log interaction for evaluation
    interaction_log = {
        "timestamp": datetime.utcnow(),
        "student_input": user_input,
        "agent_response": response,
        "learning_summary": learning_summary,
        "plan": plan,
        "tools_used": [tool.name for tool in agent.tools if
                       tool.name in plan],
        "response_length": len(response),
    }
    eval_collection.insert_one(interaction_log)

    # Update memory with learning summary
    memory.chat_memory.clear()
    memory.chat_memory.add_user_message(f"Learning progress summary:
                                        {learning_summary}")

    logger.info("Updated memory with learning summary.")

    return {"response": response, "learning_summary":
                                   learning_summary,
              "interaction_id": str(interaction_log["_id"])}

# Step 5: Define `/feedback` endpoint to collect user feedback
class FeedbackRequest(BaseModel):
    interaction_id: str
    rating: int
    comments: str = None

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    feedback_doc = {
        "interaction_id": feedback.interaction_id,
        "rating": feedback.rating,
        "comments": feedback.comments,
        "timestamp": datetime.utcnow()
    }
    feedback_collection.insert_one(feedback_doc)
    return {"message": "Thank you for your feedback!"}

# Step 6: Define root GET endpoint for health check
@app.get("/")
async def root():
    return {"message": "Personalized Learning Tutor AI is running.
                        POST your queries to /query"}

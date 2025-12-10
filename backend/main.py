from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv(override=True)

# 2. Setup FastAPI
app = FastAPI()

# 3. Enable CORS (Critical for Frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows localhost:3000 to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Initialize Gemini via OpenAI Bridge
# Check if key exists
api_key = os.getenv("GEMINI_API_KEY")
print(f"DEBUG: Loaded Key starts with: {api_key[:5] if api_key else 'None'}")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in .env")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 5. Initialize Vector DB (Mock/Memory for now)
qdrant = QdrantClient(location=":memory:")

# 6. Data Models
users_db = {} # In-memory storage

class AuthRequest(BaseModel):
    email: str
    password: str
    name: str = None

class ChatRequest(BaseModel):
    query: str
    context: str = ""

# 7. Routes
@app.get("/")
def read_root():
    return {"message": "Physical AI Backend is Running"}

@app.post("/auth/signup")
async def signup(req: AuthRequest):
    if req.email in users_db:
        return {"status": "error", "message": "User exists"}
    users_db[req.email] = req.dict()
    return {"status": "success", "user": req.dict()}

@app.post("/auth/login")
async def login(req: AuthRequest):
    user = users_db.get(req.email)
    if not user or user['password'] != req.password:
        return {"status": "error", "message": "Invalid credentials"}
    return {"status": "success", "user": user}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        print(f"Received Query: {req.query}")
        
        # Construct the Prompt
        system_prompt = "You are a specialized Physical AI & Robotics Professor. Answer the question clearly and academically."
        user_prompt = f"""
        Context from Textbook:
        {req.context}
        
        Student Question:
        {req.query}
        """

        # Call Gemini (using 2.0 Flash)
        completion = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = completion.choices[0].message.content
        return {"response": answer}

    except Exception as e:
        print(f"Error: {e}")
        return {"response": f"Error processing request: {str(e)}"}

# --- Add this Data Model ---
class TranslateRequest(BaseModel):
    text: str

class PersonalizeRequest(BaseModel):
    text: str
    context: dict

# --- Add this Endpoint ---
@app.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    try:
        # System prompt for professional translation
        system_prompt = "You are a professional technical translator. Translate the following text into Urdu. Keep technical terms (like ROS 2, Python, Node, GPU) in English. Output ONLY the Urdu translation."
        
        completion = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.text}
            ]
        )
        
        translated_text = completion.choices[0].message.content
        return {"translated_text": translated_text}

    except Exception as e:
        print(f"Translation Error: {e}")
        return {"translated_text": "Error: Could not translate."}

@app.post("/personalize")
async def personalize_endpoint(req: PersonalizeRequest):
    try:
        has_gpu = req.context.get("hasGPU", False)
        prompt_emphasis = ""
        if has_gpu:
            prompt_emphasis = "emphasize Local NVIDIA Isaac Sim."
        else:
            prompt_emphasis = "emphasize Cloud/Google Colab alternatives."

        system_prompt = f"Rewrite this technical content. If hasGPU is false, {prompt_emphasis} If true, {prompt_emphasis}"
        user_prompt = req.text

        completion = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        personalized_text = completion.choices[0].message.content
        return {"personalized_text": personalized_text}

    except Exception as e:
        print(f"Personalization Error: {e}")
        return {"personalized_text": "Error: Could not personalize."}
=======
# completed

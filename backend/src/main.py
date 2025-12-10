from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI-Native Textbook API",
    description="API for the Physical AI & Humanoid Robotics textbook platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-Native Textbook API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include API routes
from .api.chapters import router as chapters_router
from .api.chat import router as chat_router
from .api.user_progress import router as user_progress_router
from .api.translation import router as translation_router

app.include_router(chapters_router, prefix="/api/v1", tags=["chapters"])
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(user_progress_router, prefix="/api/v1", tags=["progress"])
app.include_router(translation_router, prefix="/api/v1", tags=["translation"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
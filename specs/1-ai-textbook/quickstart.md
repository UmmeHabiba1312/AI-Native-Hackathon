# Quickstart Guide: AI-Native Textbook – Physical AI & Humanoid Robotics

## Overview
This guide provides a step-by-step approach to setting up and running the AI-Native Textbook for Physical AI & Humanoid Robotics project. The project consists of a Docusaurus-based frontend textbook, a FastAPI backend for RAG chatbot functionality, and AI subagents for content generation.

## Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.11+
- Git
- Docker (for containerized services)
- Access to OpenAI API or compatible LLM service
- Qdrant vector database (local or cloud)
- Neon Postgres database (local or cloud)

## Local Development Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database URLs

# Run backend server
python -m src.main
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with backend API URL and other configurations

# Run development server
npm start
```

### 4. Vector Database Setup (Qdrant)
```bash
# Option 1: Run locally with Docker
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# Option 2: Use Qdrant Cloud
# Configure in .env with QDRANT_URL and QDRANT_API_KEY
```

### 5. Database Setup (Neon Postgres)
```bash
# Option 1: Use Neon Postgres cloud service
# Create a project at https://neon.tech
# Add the connection string to your .env file

# Option 2: Run locally with Docker
docker run --name textbook-postgres \
  -e POSTGRES_DB=textbook \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 -d postgres:15
```

## Content Generation with Subagents

### 1. Initialize Subagents
```bash
# From project root
cd subagents

# Set up subagent environment
python -m venv subagent_env
source subagent_env/bin/activate
pip install -r requirements.txt
```

### 2. Generate Textbook Content
```bash
# Generate a specific chapter
python -m subagents.chapter_writer.generate \
  --module "module1-ros2" \
  --chapter "intro" \
  --output-path "docs/module1-ros2/intro.md"

# Generate all content for a module
python -m subagents.chapter_writer.generate_module \
  --module "module1-ros2"
```

### 3. Validate Code Examples
```bash
# Validate all code examples in a chapter
python -m subagents.code_generator.validate \
  --chapter-path "docs/module1-ros2/intro.md"

# Generate and validate code snippets
python -m subagents.code_generator.generate \
  --chapter "intro" \
  --platform "ros2"
```

## Running the Full Application

### 1. Start All Services
```bash
# Terminal 1: Start Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Terminal 2: Start Postgres
docker run --name textbook-postgres -e POSTGRES_DB=textbook -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:15

# Terminal 3: Start backend
cd backend
source venv/bin/activate
python -m src.main

# Terminal 4: Start frontend
cd frontend
npm start
```

### 2. Initialize Textbook Content
```bash
# Upload textbook content to vector database
cd backend
python -m src.scripts.initialize_rag
```

## Key Features Setup

### 1. RAG Chatbot Integration
- The chatbot endpoint is available at `/api/chat`
- It connects to Qdrant for context retrieval
- Configure the model in `backend/src/config/settings.py`

### 2. Personalization Engine
- User profiles are stored in Neon Postgres
- Personalization is applied via the `PersonalizationAgent`
- Access via `/api/personalization` endpoint

### 3. Urdu Translation
- Translation service is available at `/api/translate`
- Uses the `UrduTranslator` subagent
- Toggle via translation button in the UI

## Development Commands

### Frontend Commands
```bash
# Build for production
npm run build

# Run tests
npm run test

# Lint code
npm run lint

# Preview production build
npm run serve
```

### Backend Commands
```bash
# Run tests
pytest

# Format code
black src/

# Lint code
flake8 src/

# Run with auto-reload
python -m src.main --reload
```

### Content Generation Commands
```bash
# Generate all textbook content
python -m scripts.generate_textbook

# Update RAG vectors after content changes
python -m scripts.update_rag

# Validate all code examples
python -m scripts.validate_code_examples
```

## Deployment

### 1. GitHub Pages Deployment
```bash
# Build frontend
cd frontend
npm run build

# Deploy to GitHub Pages
npm run deploy
```

### 2. Vercel Deployment
```bash
# Frontend deployment
cd frontend
vercel --prod

# Backend deployment
cd backend
vercel --prod
```

## Troubleshooting

### Common Issues
1. **Port conflicts**: Ensure ports 3000 (frontend), 8000 (backend), 5432 (Postgres), 6333 (Qdrant) are available
2. **API keys**: Verify all required API keys are set in environment variables
3. **Database connections**: Check that database URLs are correct and services are running

### Debugging Tips
- Check backend logs for API errors
- Verify vector database is populated with textbook content
- Ensure all subagents are properly configured
- Confirm RAG integration is working with test queries

## Next Steps
1. Customize the Docusaurus theme and styling
2. Add more modules and chapters
3. Enhance the chatbot with additional capabilities
4. Implement advanced personalization features
5. Add more interactive elements to the textbook
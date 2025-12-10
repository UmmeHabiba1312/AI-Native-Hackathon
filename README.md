# AI-Native Textbook for Physical AI & Humanoid Robotics

An interactive, AI-native educational platform that teaches Physical AI and Humanoid Robotics through a Docusaurus-based textbook and an integrated RAG chatbot.

## Project Overview

This project creates a complete educational platform for learning Physical AI and Humanoid Robotics concepts. It features:

- **Interactive Textbook**: Docusaurus-based textbook covering 5 modules:
  1. Robotic Nervous System (ROS 2)
  2. Digital Twin (Gazebo & Unity)
  3. AI-Robot Brain (NVIDIA Isaac)
  4. Vision-Language-Action (VLA)
  5. Capstone Project: Autonomous Humanoid

- **AI-Powered Assistance**: RAG chatbot that can answer questions using textbook content as context

- **Personalized Learning**: Content adapts based on user profile and progress

- **Multilingual Support**: Urdu translation on demand

- **Interactive Code Examples**: Execute and practice code examples directly in the textbook

## Architecture

The platform consists of:

- **Frontend**: Docusaurus-based textbook with React components for chatbot, personalization, and translation
- **Backend**: FastAPI API with RAG functionality, user management, and content services
- **Vector Database**: Qdrant for RAG chatbot context retrieval
- **Relational Database**: Neon Postgres for user data and progress tracking
- **AI Subagents**: Specialized agents for content generation, code validation, and personalization

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- Git
- Docker (for containerized services)
- Access to OpenAI API or compatible LLM service

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```

4. Create environment files:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

### Running the Application

1. Start the backend:
   ```bash
   cd backend
   source venv/bin/activate
   python -m src.main
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm start
   ```

## Development

The project uses a specification-driven development approach with the following phases:

1. **Specification** (`/specs/[feature]/spec.md`): User stories and requirements
2. **Planning** (`/specs/[feature]/plan.md`): Technical architecture and data models
3. **Tasks** (`/specs/[feature]/tasks.md`): Implementation tasks organized by user stories
4. **Implementation**: Code implementation following the task list

## Project Structure

```
├── backend/                 # FastAPI backend services
│   ├── src/
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   ├── api/            # API endpoints
│   │   ├── config/         # Configuration
│   │   └── utils/          # Utility functions
│   └── requirements.txt
├── frontend/               # Docusaurus textbook frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   └── pages/          # Page components
│   ├── docusaurus.config.ts # Docusaurus configuration
│   └── package.json
├── docs/                   # Textbook content (markdown files)
│   ├── module1-ros2/
│   ├── module2-digital-twin/
│   ├── module3-ai-brain/
│   ├── module4-vla/
│   └── capstone/
├── subagents/              # AI subagents for content generation
│   ├── chapter_writer/
│   ├── code_generator/
│   └── ...
└── skills/                 # Reusable skills for subagents
    ├── summarize_chapter/
    ├── generate_exercise/
    └── ...
```

## Features

### Textbook Content
- Structured learning modules with clear objectives
- Code examples for ROS 2, Isaac, Unity, and Python
- Diagrams and illustrations
- Hands-on exercises

### AI Integration
- RAG chatbot with contextual answers
- Text highlighting for questions
- Content personalization based on user profile

### User Experience
- Urdu translation on demand
- Progress tracking
- Responsive design for multiple devices

## Contributing

This project follows a specification-driven development approach. To contribute:

1. Review the existing specifications in `/specs/`
2. Follow the task lists in `/specs/[feature]/tasks.md`
3. Ensure new code follows the architectural patterns established in the plan
4. Write tests for new functionality
5. Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# Implementation Plan: AI-Native Textbook – Physical AI & Humanoid Robotics

**Branch**: `1-ai-textbook` | **Date**: 2025-12-09 | **Spec**: [link to spec.md](../spec.md)
**Input**: Feature specification from `/specs/1-ai-textbook/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a Docusaurus-based interactive textbook covering Physical AI and Humanoid Robotics with 5 modules (Robotic Nervous System, Digital Twin, AI-Robot Brain, Vision-Language-Action, and Capstone Project). The textbook will integrate an RAG chatbot, support personalization, Urdu translation, and include interactive code examples. Implementation will use subagents (ChapterWriter, CodeSnippetGenerator, DiagramAssistant, etc.) and skills (ValidateCode, GenerateExercise, etc.) for content generation and management.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend
**Primary Dependencies**: Docusaurus, FastAPI, Qdrant, Neon Postgres, React
**Storage**: Markdown files for content, Qdrant for vector storage, Neon Postgres for user data
**Testing**: pytest for backend, Jest for frontend
**Target Platform**: Web-based (GitHub Pages/Vercel deployment)
**Project Type**: Web application (frontend + backend)
**Performance Goals**: Page load under 3 seconds, chatbot response under 3 seconds, support 1000 concurrent users
**Constraints**: <200ms p95 for content retrieval, mobile-responsive design, offline-capable content caching
**Scale/Scope**: 5 course modules, 15+ chapters, 1000+ learners, 100+ code examples

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Frontend Stack: Docusaurus for the textbook with TypeScript and React for interactive components and chatbot integration
- Backend Stack: FastAPI with Python for the RAG chatbot API, integrated with Neon Postgres for relational data and Qdrant for vector storage
- Authentication: Better-Auth to support user profiles and personalized learning experiences
- Robotics Target: Physical AI and Humanoid Robotics concepts, with support for ROS 2, Gazebo/Unity simulations, and NVIDIA Isaac Sim
- Development Workflow: Spec-Kit Plus workflow with AI-native workflows using Claude Code and specification-driven development

## Project Structure

### Documentation (this feature)

```text
specs/1-ai-textbook/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
docs/
├── module1-ros2/
│   ├── intro.md
│   ├── nodes-topics.md
│   └── urdf-python.md
├── module2-digital-twin/
│   ├── gazebo-simulation.md
│   ├── sensor-simulation.md
│   └── unity-visualization.md
├── module3-ai-brain/
│   ├── isaac-sim.md
│   ├── isaac-ros.md
│   └── nav2-planning.md
├── module4-vla/
│   ├── voice-to-action.md
│   ├── cognitive-planning.md
│   └── multi-modal.md
└── capstone/
    └── autonomous-humanoid.md

frontend/
├── src/
│   ├── components/
│   │   ├── TextbookViewer/
│   │   ├── Chatbot/
│   │   ├── Personalization/
│   │   └── Translation/
│   ├── pages/
│   ├── services/
│   └── hooks/
├── static/
└── docusaurus.config.js

backend/
├── src/
│   ├── models/
│   ├── services/
│   ├── api/
│   └── agents/
├── tests/
└── requirements.txt

subagents/
├── chapter_writer/
├── code_generator/
├── diagram_assistant/
└── personalization_agent/

skills/
├── summarize_chapter/
├── generate_exercise/
├── validate_code/
└── update_rag/
```

**Structure Decision**: Web application structure with separate frontend (Docusaurus-based textbook) and backend (FastAPI API for RAG chatbot and user services). Content is stored in markdown files under docs/ following the module/chapter structure. Subagents and skills are organized in separate directories to support the AI-native content generation workflow.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple project structure | Required for separation of concerns between textbook frontend, backend services, and AI subagents | Single project would create monolithic codebase difficult to maintain and extend |
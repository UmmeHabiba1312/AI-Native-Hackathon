---
id: 20251209-0005
title: Create AI Textbook Plan
stage: plan
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: ai-textbook
branch: 1-ai-textbook
user: FC
command: /sp.plan
labels: [planning, architecture, textbook, physical-ai, docusaurus]
links:
  spec: specs/1-ai-textbook/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-ai-textbook/plan.md
 - specs/1-ai-textbook/research.md
 - specs/1-ai-textbook/data-model.md
 - specs/1-ai-textbook/quickstart.md
 - specs/1-ai-textbook/contracts/textbook-api.yaml
 - history/prompts/ai-textbook/20251209-0005-create-ai-textbook-plan.plan.prompt.md
tests:
 - null
---

## Prompt

/sp,plan Project: AI-Native Textbook – Physical AI & Humanoid Robotics

Purpose:
Create a structured plan to generate the complete textbook covering 4 modules with chapters, including subagent and skill assignments for Claude Code implementation.

---

### Module 1: Robotic Nervous System (ROS 2)
Focus: Middleware for robot control and humanoid description
Chapters:
1. Introduction to ROS 2 and Robotic Nervous System
   - Subagents: ChapterWriter, ContentReviewer
   - Skills: SummarizeChapter, PersonalizeContent
2. ROS 2 Nodes, Topics, Services, and Actions
   - Subagents: ChapterWriter, CodeSnippetGenerator, ContentReviewer
   - Skills: ValidateCode, GenerateExercise
3. URDF & Python Agent Integration
   - Subagents: ChapterWriter, CodeSnippetGenerator, DiagramAssistant
   - Skills: GenerateDiagrams, ValidateCode

---

### Module 2: Digital Twin (Gazebo & Unity)
Focus: Simulation and environment building
Chapters:
1. Gazebo Physics Simulation & Environment Setup
   - Subagents: ChapterWriter, CodeSnippetGenerator, ContentReviewer
   - Skills: ValidateCode, GenerateExercise
2. Sensor Simulation: LiDAR, IMU, Depth Cameras
   - Subagents: ChapterWriter, CodeSnippetGenerator, DiagramAssistant
   - Skills: GenerateDiagrams, ValidateCode
3. Unity Visualization & Human-Robot Interaction
   - Subagents: ChapterWriter, CodeSnippetGenerator, DiagramAssistant
   - Skills: ValidateCode, GenerateDiagrams

---

### Module 3: AI-Robot Brain (NVIDIA Isaac)
Focus: Advanced perception, navigation, and AI integration
Chapters:
1. NVIDIA Isaac Sim & Synthetic Data Generation
   - Subagents: ChapterWriter, CodeSnippetGenerator
   - Skills: ValidateCode, GenerateExercise
2. Isaac ROS: Hardware-Accelerated VSLAM & Navigation
   - Subagents: ChapterWriter, CodeSnippetGenerator, DiagramAssistant
   - Skills: ValidateCode, GenerateDiagrams
3. Nav2 & Path Planning for Humanoid Movement
   - Subagents: ChapterWriter, CodeSnippetGenerator
   - Skills: ValidateCode, GenerateExercise

---

### Module 4: Vision-Language-Action (VLA)
Focus: Convergence of LLMs and robotics
Chapters:
1. Voice-to-Action: Using Whisper for Commands
   - Subagents: ChapterWriter, CodeSnippetGenerator
   - Skills: ValidateCode, GenerateExercise
2. Cognitive Planning: Translating Commands into Actions
   - Subagents: ChapterWriter, CodeSnippetGenerator, DiagramAssistant
   - Skills: ValidateCode, GenerateDiagrams
3. Multi-modal Interaction: Speech, Vision, Gesture
   - Subagents: ChapterWriter, CodeSnippetGenerator, DiagramAssistant
   - Skills: ValidateCode, GenerateDiagrams

---

### Capstone Integration Chapter
Chapter: Autonomous Humanoid Project
- Subagents: ChapterWriter, CodeSnippetGenerator, DiagramAssistant, QuizGenerator, ContentReviewer
- Skills: ValidateCode, GenerateDiagrams, GenerateExercise, CheckDependencies

---

### Notes:
- Each chapter will have learning objectives, examples, exercises, and diagrams.
- Subagents collaborate: ChapterWriter generates content → CodeSnippetGenerator adds examples → DiagramAssistant adds visuals → ContentReviewer ensures clarity.
- Skills are reused across chapters for consistency.
- Deployment and RAG integration will be added later.

End of Plan.

## Response snapshot

Created a comprehensive implementation plan for the AI-Native Textbook for Physical AI & Humanoid Robotics project. The plan includes:

1. **Implementation Plan** (plan.md): Detailed technical approach with project structure, constitution check, and complexity tracking
2. **Research** (research.md): Technology landscape analysis, architecture patterns, subagent architecture, integration points, deployment strategy, security considerations, performance optimization, and risk mitigation
3. **Data Model** (data-model.md): Complete schema for all entities including Chapter, Module, User, UserProgress, CodeExample, Diagram, and Exercise with relationships and validation rules
4. **Quickstart Guide** (quickstart.md): Step-by-step setup instructions for local development, backend/frontend setup, database configuration, content generation, and deployment
5. **API Contract** (contracts/textbook-api.yaml): OpenAPI specification for the textbook platform API with endpoints for chapters, chat, user progress, and translation

The plan follows the project constitution requirements with Docusaurus frontend, FastAPI backend, Qdrant vector database, and Neon Postgres. It incorporates the subagent architecture mentioned in the requirements with ChapterWriter, CodeSnippetGenerator, DiagramAssistant, and other specialized agents.

## Outcome

- ✅ Impact: Comprehensive planning completed for AI textbook project with all required artifacts
- 🧪 Tests: API contract defined for testing endpoints
- 📁 Files: Created plan.md, research.md, data-model.md, quickstart.md, and API contract
- 🔁 Next prompts: Ready to proceed with /sp.tasks for implementation tasks
- 🧠 Reflection: Plan addresses all modules and chapters with proper technical architecture

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Proceed with task generation
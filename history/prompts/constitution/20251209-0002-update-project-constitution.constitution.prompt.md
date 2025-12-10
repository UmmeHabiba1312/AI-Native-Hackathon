---
id: 20251209-0002
title: Update Project Constitution
stage: constitution
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: FC
command: /sp.constitution
labels: [constitution, governance, project-setup]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
 - history/prompts/constitution/20251209-0002-update-project-constitution.constitution.prompt.md
tests:
 - null
---

## Prompt

""" Project Name:
AI-Native Textbook & RAG Chatbot for Physical AI and Humanoid Robotics

Core Purpose:
To create a fully AI-native, interactive, and scalable educational platform that teaches Physical AI and Humanoid Robotics through a Docusaurus-based textbook and an integrated RAG chatbot powered by OpenAI Agents/ChatKit, FastAPI, Neon Postgres, and Qdrant.

Project Vision:
Deliver a world-class learning experience where theory, simulation, coding, and robotics practice come together in a single intelligent platform. Learners should be able to read the textbook, ask questions, run examples, and get personalized guidance through an embedded chatbot.

Guiding Values:
- Specification-driven development (clarity → design → tasks → implementation)
- High modularity, readability, and maintainability
- AI-native workflows using Spec-Kit Plus and Claude Code
- Transparency in architectural, design, and decision history
- Reproducibility and testability
- Safety, reliability, and future extensibility

High-level Objectives:
1. Write a complete Docusaurus-based textbook covering:
   - Physical AI foundations
   - Humanoid robotics fundamentals
   - ROS 2, Gazebo/Unity simulations
   - NVIDIA Isaac Sim & RL/LLM-driven control
   - Vision-Language-Action robotics
   - Hands-on humanoid projects
2. Build an intelligent RAG chatbot that can:
   - Answer questions contextually using the textbook
   - Use a vector database (Qdrant) and relational DB (Neon)
   - Use OpenAI Agents/ChatKit for conversation and tool calling
3. Implement optional (bonus) features:
   - Claude Subagent / Skills extensions
   - User signup → personalized learning profiles
   - Urdu translation mode
4. Ensure a clean, modular repo using Claude Code for multi-file coding tasks.

Constitutional Principles for the AI System:
- Follow the specification-first workflow of Spec-Kit Plus:
  - `/sp.specify` → formulate structured requirements
  - `/sp.plan` → break into architecture & components
  - `/sp.tasks` → define actionable, testable tasks
  - `/sp.implement` → generate and refine code
- All agents maintain consistency with the project vision and objectives.
- All generated content must be well organized, commented, and maintainable.
- No shortcuts: follow clarity, modularity, and rationale in every decision.

Primary Deliverables:
- Complete textbook in Docusaurus format
- Fully working RAG chatbot embedded into the book
- GitHub repo with clean architecture
- Demo video and deployment links

Non-Goals:
- Building physical robots
- Running non-open source or insecure tools
- Implementing advanced hardware-only robotics unless simulated

Project Success Criteria:
- Stable deployment
- Textbook quality meets academic standards
- Chatbot answers reliably and contextually
- Architecture remains clean and extensible
- Bonus features implemented where feasible

End of Constitution.
 """

## Response snapshot

Updated the project constitution at `.specify/memory/constitution.md` to reflect the specific requirements for the AI-Native Textbook & RAG Chatbot for Physical AI and Humanoid Robotics project. The changes included:

1. Updated the project title to "AI-Native Textbook & RAG Chatbot Constitution"
2. Revised all five core principles to align with the project:
   - I. Frontend Stack: Updated to reflect Docusaurus textbook focus with React for chatbot integration
   - II. Backend Stack: Updated to reflect RAG chatbot architecture with Neon Postgres and Qdrant
   - III. Authentication: Updated to reflect user profile features
   - IV. Robotics Target: Updated to reflect Physical AI focus with ROS 2, Gazebo/Unity, and Isaac Sim
   - V. Development Workflow: Updated to reflect AI-native workflow with Claude Code
3. Enhanced the Governance section to include the project's guiding values
4. Updated the version from 1.0.0 to 1.1.0 with today's date as the amendment date
5. Updated the sync impact report to reflect all changes made

## Outcome

- ✅ Impact: Updated project constitution to align with Physical AI and Humanoid Robotics textbook project
- 🧪 Tests: No tests required for constitution update
- 📁 Files: Updated .specify/memory/constitution.md and created PHR record
- 🔁 Next prompts: Ready to proceed with specification, planning, and implementation phases
- 🧠 Reflection: Constitution now properly reflects the project's specific requirements and guiding principles

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Proceed with /sp.specify to create project specification
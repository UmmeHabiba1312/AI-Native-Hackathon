---
id: 20251209-0004
title: Create AI Textbook Specification
stage: spec
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: ai-textbook
branch: 1-ai-textbook
user: FC
command: /sp.specify
labels: [specification, textbook, physical-ai, humanoid-robotics, docusaurus]
links:
  spec: specs/1-ai-textbook/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-ai-textbook/spec.md
 - specs/1-ai-textbook/checklists/requirements.md
 - history/prompts/ai-textbook/20251209-0004-create-ai-textbook-spec.spec.prompt.md
tests:
 - null
---

## Prompt

Component: AI-Native Textbook for Physical AI & Humanoid Robotics

Purpose:
Generate a complete, interactive, AI-native textbook covering Physical AI and Humanoid Robotics, fully integrated with RAG chatbot and MCP server for content retrieval and personalization.

---

Functional Requirements:

1. **Book Structure**
   - Chapters mapped to course modules:
     1. Robotic Nervous System (ROS 2)
     2. Digital Twin (Gazebo & Unity)
     3. AI-Robot Brain (NVIDIA Isaac)
     4. Vision-Language-Action (VLA)
     5. Capstone Project: Autonomous Humanoid
   - Each chapter includes:
     - Learning objectives
     - Step-by-step explanations
     - Code examples (ROS 2, Isaac, Unity, Python)
     - Diagrams and illustrations
     - Hands-on exercises
     - Optional translation button (Urdu)
     - Personalization button (based on user profile)

2. **Integration**
   - Connects to **MCP server** for:
     - Storing/retrieving chapter content
     - Enabling RAG chatbot access to chapter-specific context
     - Synchronizing personalized content per user
   - Connects to **RAG backend** (Qdrant + Neon Postgres) for query handling.

3. **Interactivity**
   - User can highlight text and ask questions via embedded chatbot
   - Personalized content adapts based on:
     - Software/hardware background
     - Chapter progress
   - Buttons for translation (Urdu) and content personalization
   - Inline code examples executable (or copy-ready)

4. **Technical Requirements**
   - Docusaurus framework
   - Markdown files organized per module:
     `/docs/module/chapter.md`
   - Metadata per chapter: title, module, difficulty, estimated time
   - Deployment: GitHub Pages or Vercel
   - Compatible with Claude Code multi-file editing

---

### Subagents for Textbook Generation

| Subagent Name       | Responsibility                                                                 |
|--------------------|-------------------------------------------------------------------------------|
| `ChapterWriter`     | Generates chapter text based on learning objectives, exercises, and examples |
| `CodeSnippetGenerator` | Generates and validates ROS 2, Isaac, Unity, Python code snippets           |
| `DiagramAssistant`  | Creates illustrations, flowcharts, and URDF diagrams                          |
| `PersonalizationAgent` | Adjusts content for logged-in user based on profile                         |
| `UrduTranslator`    | Translates chapters to Urdu on demand                                         |
| `RAGIntegrator`     | Embeds chapter content into RAG vector database for contextual queries       |
| `QuizGenerator`     | Creates exercises, coding tasks, and quizzes                                  |
| `ContentReviewer`   | Ensures chapter clarity, formatting, and consistency                         |
| `DeploymentAgent`   | Handles Docusaurus build and deployment                                       |

---

### Skills for Textbook Generation

| Skill Name            | Usage                                                                 |
|-----------------------|----------------------------------------------------------------------|
| `SummarizeChapter`     | Produces concise chapter summaries for review or chatbot             |
| `GenerateExercise`     | Generates coding exercises, problem sets, or quizzes                 |
| `ValidateCode`         | Checks code snippets for syntax, ROS 2, Isaac, and Unity compatibility |
| `UpdateRAG`            | Pushes chapter content to vector DB for RAG chatbot queries          |
| `AnswerWithContext`    | Retrieves relevant chapter text for chatbot responses                |
| `PersonalizeContent`   | Adjusts chapter text dynamically according to user profile           |
| `TranslateText`        | Translates text into Urdu (or other languages)                       |
| `GenerateDiagrams`     | Creates structured diagrams from textual descriptions                |
| `CheckDependencies`    | Ensures all code, diagrams, and references are available            |
| `DeploySite`           | Automates Docusaurus build and deployment workflow                   |

---

### Acceptance Criteria:

- All chapters generated with correct structure and code examples
- Chatbot can answer questions using RAG + MCP server context
- Personalization and Urdu translation buttons work
- Docusaurus site deployed publicly on GitHub Pages/Vercel
- Code and diagrams are validated and consistent across chapters
- Subagents and Skills work together to produce a maintainable, modular book

---

## Response snapshot

Created a comprehensive specification for the AI-Native Textbook for Physical AI & Humanoid Robotics project. The specification includes:

1. Five prioritized user stories covering:
   - Access to interactive textbook content (P1)
   - RAG chatbot integration for contextual answers (P1)
   - Personalized learning experience (P2)
   - Urdu translation functionality (P2)
   - Interactive code examples (P3)

2. 18 functional requirements covering:
   - Docusaurus-based web interface
   - Content display with learning objectives and code examples
   - RAG chatbot integration
   - MCP server connectivity
   - User authentication and personalization
   - Urdu translation
   - Code validation and execution
   - Progress tracking
   - Deployment capabilities

3. 10 measurable success criteria including:
   - Content accessibility and navigation performance
   - Chatbot effectiveness
   - User engagement with personalization
   - Multilingual accessibility usage
   - Code example success rate
   - System uptime and concurrency support

4. Key entities defined: Chapter, Module, User Profile, Code Example, Diagram, Translation, Chatbot Query, and Exercise

5. Quality checklist confirming all requirements are met

## Outcome

- ✅ Impact: Created comprehensive specification for AI-native textbook project
- 🧪 Tests: Specification includes testable requirements and acceptance scenarios
- 📁 Files: Created spec.md and requirements checklist in specs/1-ai-textbook/
- 🔁 Next prompts: Ready to proceed with /sp.plan for architectural planning
- 🧠 Reflection: Specification captures all essential elements of the AI-native textbook with Physical AI focus

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Proceed with architectural planning
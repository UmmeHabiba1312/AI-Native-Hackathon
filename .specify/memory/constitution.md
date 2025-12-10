<!--
Version change: 1.0.0 -> 1.1.0
Modified principles:
  - I. Frontend Stack: Updated to reflect Docusaurus textbook focus
  - II. Backend Stack: Updated to reflect RAG chatbot architecture
  - III. Authentication: Updated to reflect user profile features
  - IV. Robotics Target: Updated to reflect Physical AI focus
  - V. Development Workflow: Updated to reflect AI-native workflow
Added sections: None
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md: ⚠ pending
  - .specify/templates/spec-template.md: ⚠ pending
  - .specify/templates/tasks-template.md: ⚠ pending
  - .specify/templates/commands/*.md: ⚠ pending
Follow-up TODOs: None
-->
# AI-Native Textbook & RAG Chatbot Constitution

## Core Principles

### I. Frontend Stack
Frontend development MUST use Docusaurus for the textbook with TypeScript and React for interactive components and chatbot integration.

### II. Backend Stack
Backend development MUST use FastAPI with Python for the RAG chatbot API, integrated with Neon Postgres for relational data and Qdrant for vector storage.

### III. Authentication
Authentication MUST be implemented using Better-Auth to support user profiles and personalized learning experiences.

### IV. Robotics Target
All robotic code and simulations MUST target Physical AI and Humanoid Robotics concepts, with support for ROS 2, Gazebo/Unity simulations, and NVIDIA Isaac Sim.

### V. Development Workflow
Development MUST follow the Spec-Kit Plus workflow: Specify -> Plan -> Task -> Implement, with AI-native workflows using Claude Code and specification-driven development.


## Governance

The Constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan. All PRs/reviews must verify compliance. Complexity must be justified. All generated content must be well organized, commented, and maintainable following the project's guiding values of specification-driven development, high modularity, readability, maintainability, AI-native workflows, transparency, reproducibility, testability, safety, reliability, and future extensibility.

**Version**: 1.1.0 | **Ratified**: 2025-11-29 | **Last Amended**: 2025-12-09

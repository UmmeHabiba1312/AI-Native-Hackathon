# ADR-1: Update Project Constitution

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-09
- **Feature:** AI-Native Textbook & RAG Chatbot for Physical AI and Humanoid Robotics
- **Context:** The project requires a clear constitution that defines the core principles and governance for developing an AI-Native Textbook & RAG Chatbot for Physical AI and Humanoid Robotics. The original constitution needed to be updated to reflect the specific requirements and technologies for this Physical AI educational platform.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

Update the project constitution to align with the AI-Native Textbook & RAG Chatbot for Physical AI and Humanoid Robotics project requirements. The updated constitution includes:

- Frontend Stack: Docusaurus for the textbook with TypeScript and React for interactive components and chatbot integration
- Backend Stack: FastAPI with Python for the RAG chatbot API, integrated with Neon Postgres for relational data and Qdrant for vector storage
- Authentication: Better-Auth to support user profiles and personalized learning experiences
- Robotics Target: Physical AI and Humanoid Robotics concepts, with support for ROS 2, Gazebo/Unity simulations, and NVIDIA Isaac Sim
- Development Workflow: Spec-Kit Plus workflow with AI-native workflows using Claude Code and specification-driven development

<!-- For technology stacks, list all components:
     - Framework: Next.js 14 (App Router)
     - Styling: Tailwind CSS v3
     - Deployment: Vercel
     - State Management: React Context (start simple)
-->

## Consequences

### Positive

- Clear guidance for development teams on technology choices and architectural patterns
- Alignment of all project activities with Physical AI and Humanoid Robotics educational goals
- Standardized approach to frontend, backend, and development workflow decisions
- Support for RAG chatbot integration and personalized learning experiences
- Foundation for scalable and maintainable educational platform

<!-- Example: Integrated tooling, excellent DX, fast deploys, strong TypeScript support -->

### Negative

- Commitment to specific technology stack may limit flexibility to adopt alternative solutions later
- Learning curve for team members unfamiliar with Docusaurus, FastAPI, or Physical AI concepts
- Potential vendor dependencies on specific services (Neon, Qdrant)

<!-- Example: Vendor lock-in to Vercel, framework coupling, learning curve -->

## Alternatives Considered

Alternative Constitution Approaches:
- Keep original constitution with minimal changes - Rejected because it didn't reflect the specific Physical AI and Humanoid Robotics focus
- Create completely new constitution from scratch - Rejected because it would lose existing governance structure
- Generic educational platform constitution - Rejected because it wouldn't address the specialized Physical AI and RAG chatbot requirements

<!-- Group alternatives by cluster:
     Alternative Stack A: Remix + styled-components + Cloudflare
     Alternative Stack B: Vite + vanilla CSS + AWS Amplify
     Why rejected: Less integrated, more setup complexity
-->

## References

- Feature Spec: N/A (Constitution update)
- Implementation Plan: .specify/memory/constitution.md
- Related ADRs: None
- Evaluator Evidence: history/prompts/constitution/20251209-0002-update-project-constitution.constitution.prompt.md
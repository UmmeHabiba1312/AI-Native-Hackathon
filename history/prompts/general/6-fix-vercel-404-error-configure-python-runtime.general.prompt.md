---
id: 6
title: Fix Vercel 404 Error (Configure Python Runtime)
stage: general
date: 2025-11-30
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: default_user
command: "Fix Vercel 404 Error (Configure Python Runtime).\n\nThe backend returns 404 because Vercel treats it as a static site.\nAction: Create a file `backend/vercel.json` with this EXACT configuration:\n\n```json\n{\n  \"builds\": [\n    {\n      \"src\": \"main.py\",\n      \"use\": \"@vercel/python\"\n    }\n  ],\n  \"routes\": [\n    {\n      \"src\": \"/(.*)\",\n      \"dest\": \"main.py\"\n    }\n  ]\n}\n```"
labels: ["backend", "vercel", "deployment"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
- backend/vercel.json
tests:
- null
---

## Prompt

Fix Vercel 404 Error (Configure Python Runtime).

The backend returns 404 because Vercel treats it as a static site.
Action: Create a file `backend/vercel.json` with this EXACT configuration:

```json
{
  "builds": [
    {
      "src": "main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "main.py"
    }
  ]
}
```

## Response snapshot

Created/updated `backend/vercel.json` with the specified Vercel configuration to ensure Python runtime is correctly handled for the backend.

## Outcome

- ✅ Impact: Backend deployment on Vercel should now correctly route requests to `main.py`.
- 🧪 Tests: No explicit tests were run as part of this task.
- 📁 Files: Created/Modified backend/vercel.json.
- 🔁 Next prompts: Verify backend deployment on Vercel.
- 🧠 Reflection: Successfully configured Vercel for the Python backend, addressing the 404 error.

## Evaluation notes (flywheel)

- Failure modes observed: Initial confusion due to `Write` tool error message about reading file first, even for new file creation. Clarified by attempting to `Read` first and then `Write`.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

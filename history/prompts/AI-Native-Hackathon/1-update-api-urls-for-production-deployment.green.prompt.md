---
ID: "1"
TITLE: "Update API URLs for Production Deployment"
STAGE: "green"
DATE_ISO: "2025-11-30"
SURFACE: "agent"
MODEL: "claude-sonnet-4-5-20250929"
FEATURE: "AI-Native-Hackathon"
BRANCH: "main"
USER: "user"
COMMAND: "Update API URLs for Production Deployment"
LABELS: ["api", "deployment", "frontend"]
LINKS:
  SPEC: "null"
  TICKET: "null"
  ADR: "null"
  PR: "null"
FILES_YAML:
  - frontend/src/lib/auth-client.ts
  - frontend/src/components/TranslateButton.tsx
  - frontend/src/components/ChatWidget.tsx
  - frontend/src/pages/login.tsx
  - frontend/docusaurus.config.ts
TESTS_YAML: []
---

### Prompt:

```
Task: Update API URLs for Production Deployment.

I have deployed the backend to Vercel: https://ai-native-hackathon-backend.vercel.app
I have deployed the frontend to GitHub Pages: https://ummehabiba1312.github.io/AI-Native-Hackathon/

Action: Find and replace all instances of `http://localhost:8000` with the new backend URL in the frontend code.

Specific files to update:

1.  **`frontend/src/lib/auth-client.ts`**
    * Update `baseURL` to: `"https://ai-native-hackathon-backend.vercel.app"`

2.  **`frontend/src/components/TranslateButton.tsx`**
    * Update the `fetch` URL to: `"https://ai-native-hackathon-backend.vercel.app/translate"`

3.  **`frontend/src/components/ChatWidget.tsx`**
    * Update the `fetch` URL to: `"https://ai-native-hackathon-backend.vercel.app/chat"`

4.  **`frontend/src/pages/login.tsx`**
    * Update the `fetch` URL to: `"https://ai-native-hackathon-backend.vercel.app/auth/${endpoint}"`

5.  **`frontend/docusaurus.config.ts`**
    * Ensure the Login button points to the internal route `to: '/login'` (do not use the backend URL here).please proceed
```

### Response:

Updated API URLs in frontend files as requested.

### Outcome and Evaluation:
The API URLs were successfully updated or verified in the specified files:
- `frontend/src/lib/auth-client.ts`: Verified existing update.
- `frontend/src/components/TranslateButton.tsx`: Updated fetch URL.
- `frontend/src/components/ChatWidget.tsx`: Updated fetch URL.
- `frontend/src/pages/login.tsx`: Verified no direct changes needed as it uses `auth-client.ts`.
- `frontend/docusaurus.config.ts`: Verified Login button URL.

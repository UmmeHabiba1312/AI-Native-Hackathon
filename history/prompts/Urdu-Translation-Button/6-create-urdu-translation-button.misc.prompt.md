---
id: 6
title: Create Urdu Translation Button
stage: misc
date: 2025-11-29
surface: agent
model: claude-sonnet-4-5-20250929
feature: Urdu-Translation-Button
branch: main
user: unknown
command: 'Task: Create Urdu Translation Button.

1.  **Create Component:** `frontend/src/components/TranslateButton.tsx`.
    * Code:
        ```tsx
        import React, { useState } from 'react';

        export default function TranslateButton() {
          const [loading, setLoading] = useState(false);

          const handleTranslate = async () => {
            setLoading(true);
            // 1. Find the main content (Docusaurus uses 'markdown' class usually)
            const contentDiv = document.querySelector('.markdown') || document.querySelector('article');

            if (!contentDiv) return;

            const originalText = contentDiv.innerText;

            try {
              const res = await fetch('http://localhost:8000/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: originalText.substring(0, 2000) }) // Limit length for speed
              });

              const data = await res.json();
              if (data.translated_text) {
                // Replace content with Urdu
                contentDiv.innerHTML = `<div dir="rtl" style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.2rem; line-height: 2;">${data.translated_text}</div>`;
              }
            } catch (e) {
              alert("Translation Failed: Is the Backend running?");
            } finally {
              setLoading(false);
            }
          };

          return (
            <button
              onClick={handleTranslate}
              style={{
                position: 'fixed', bottom: '100px', right: '20px',
                padding: '10px 20px', backgroundColor: '#25c2a0', color: 'white',
                border: 'none', borderRadius: '5px', cursor: 'pointer', zIndex: 200
              }}
            >
              {loading ? 'Translating...' : 'اردو / Translate'}
            </button>
          );
        }
        ```

2.  **Register:** Update `frontend/src/theme/Root.tsx` to include `<TranslateButton />` inside the Provider, just like you did for the ChatWidget.'
labels: ["react", "docusaurus", "translation"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - frontend/src/components/TranslateButton.tsx
  - frontend/src/theme/Root.tsx
tests:

---

## Prompt

Task: Create Urdu Translation Button.

1.  **Create Component:** `frontend/src/components/TranslateButton.tsx`.
    * Code:
        ```tsx
        import React, { useState } from 'react';

        export default function TranslateButton() {
          const [loading, setLoading] = useState(false);

          const handleTranslate = async () => {
            setLoading(true);
            // 1. Find the main content (Docusaurus uses 'markdown' class usually)
            const contentDiv = document.querySelector('.markdown') || document.querySelector('article');

            if (!contentDiv) return;

            const originalText = contentDiv.innerText;

            try {
              const res = await fetch('http://localhost:8000/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: originalText.substring(0, 2000) }) // Limit length for speed
              });

              const data = await res.json();
              if (data.translated_text) {
                // Replace content with Urdu
                contentDiv.innerHTML = `<div dir="rtl" style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.2rem; line-height: 2;">${data.translated_text}</div>`;
              }
            } catch (e) {
              alert("Translation Failed: Is the Backend running?");
            } finally {
              setLoading(false);
            }
          };

          return (
            <button
              onClick={handleTranslate}
              style={{
                position: 'fixed', bottom: '100px', right: '20px',
                padding: '10px 20px', backgroundColor: '#25c2a0', color: 'white',
                border: 'none', borderRadius: '5px', cursor: 'pointer', zIndex: 200
              }}
            >
              {loading ? 'Translating...' : 'اردو / Translate'}
            </button>
          );
        }
        ```

2.  **Register:** Update `frontend/src/theme/Root.tsx` to include `<TranslateButton />` inside the Provider, just like you did for the ChatWidget.

## Response snapshot

I have created the `frontend/src/components/TranslateButton.tsx` component and registered it in `frontend/src/theme/Root.tsx`.

## Outcome

- ✅ Impact: Created and registered the Urdu translation button component.
- 🧪 Tests: N/A
- 📁 Files: `frontend/src/components/TranslateButton.tsx` created, `frontend/src/theme/Root.tsx` modified.
- 🔁 Next prompts: Awaiting further instructions from the user.
- 🧠 Reflection: Successfully implemented the translation button feature as requested.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

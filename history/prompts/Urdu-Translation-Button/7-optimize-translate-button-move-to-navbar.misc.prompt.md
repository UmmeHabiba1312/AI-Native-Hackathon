---
id: 7
title: Optimize Translate Button & Move to Navbar
stage: misc
date: 2025-11-29
surface: agent
model: claude-sonnet-4-5-20250929
feature: Urdu-Translation-Button
branch: main
user: unknown
command: 'Task: Optimize Translate Button & Move to Navbar

Action: Rewrite "frontend/src/components/TranslateButton.tsx" with this code:

```tsx
import React, { useState } from 'react';

export default function TranslateButton() {
  const [loading, setLoading] = useState(false);

  const handleTranslate = async () => {
    if (loading) return;
    setLoading(true);

    // 1. Find the main content
    const contentDiv = document.querySelector('.markdown') || document.querySelector('article');
    if (!contentDiv) {
      setLoading(false);
      return;
    }

    const originalText = contentDiv.innerText;

    // 2. SPEED FIX: Only translate first 1500 chars for the demo
    const textToSend = originalText.substring(0, 1500) + "... (Truncated for Demo)";

    try {
      const res = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToSend })
      });

      const data = await res.json();
      if (data.translated_text) {
        // Replace content with Urdu
        contentDiv.innerHTML = `<div dir="rtl" style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.2rem; line-height: 2; padding: 20px;">${data.translated_text}</div>`;
      }
    } catch (e) {
      console.error(e);
      alert("Backend not connected!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="navbar__item">
      <button
        onClick={handleTranslate}
        className="clean-btn navbar__link"
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', fontSize: '1.2rem' }}
        title="Translate to Urdu"
      >
        {loading ? (
          <span className="navbar-icon">⏳</span>
        ) : (
          /* Google Translate Icon style */
          <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
          </svg>
        )}
      </button>
    </div>
  );
}
```'
labels: ["react", "docusaurus", "translation", "optimization"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - frontend/src/components/TranslateButton.tsx
tests:

---

## Prompt

Task: Optimize Translate Button & Move to Navbar

Action: Rewrite "frontend/src/components/TranslateButton.tsx" with this code:

```tsx
import React, { useState } from 'react';

export default function TranslateButton() {
  const [loading, setLoading] = useState(false);

  const handleTranslate = async () => {
    if (loading) return;
    setLoading(true);

    // 1. Find the main content
    const contentDiv = document.querySelector('.markdown') || document.querySelector('article');
    if (!contentDiv) {
      setLoading(false);
      return;
    }

    const originalText = contentDiv.innerText;

    // 2. SPEED FIX: Only translate first 1500 chars for the demo
    const textToSend = originalText.substring(0, 1500) + "... (Truncated for Demo)";

    try {
      const res = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToSend })
      });

      const data = await res.json();
      if (data.translated_text) {
        // Replace content with Urdu
        contentDiv.innerHTML = `<div dir="rtl" style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.2rem; line-height: 2; padding: 20px;">${data.translated_text}</div>`;
      }
    } catch (e) {
      console.error(e);
      alert("Backend not connected!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="navbar__item">
      <button
        onClick={handleTranslate}
        className="clean-btn navbar__link"
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', fontSize: '1.2rem' }}
        title="Translate to Urdu"
      >
        {loading ? (
          <span className="navbar-icon">⏳</span>
        ) : (
          /* Google Translate Icon style */
          <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
          </svg>
        )}
      </button>
    </div>
  );
}

## Response snapshot

I have updated the `frontend/src/components/TranslateButton.tsx` file with the optimized code and moved its styling to be compatible with the Docusaurus navbar.

## Outcome

- ✅ Impact: Optimized translation button and adapted it for navbar integration.
- 🧪 Tests: N/A
- 📁 Files: `frontend/src/components/TranslateButton.tsx` modified.
- 🔁 Next prompts: Awaiting further instructions from the user, specifically regarding integrating into the navbar configuration.
- 🧠 Reflection: Successfully implemented the requested optimizations and styling changes for the translation button.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

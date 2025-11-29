---
id: 12
title: Upgrade Translate Button to Dropdown Menu
stage: misc
date: 2025-11-29
surface: agent
model: claude-sonnet-4-5-20250929
feature: Urdu-Translation-Button
branch: main
user: unknown
command: 'Task: Upgrade Translate Button to Dropdown Menu

Action: Rewrite `frontend/src/components/TranslateButton.tsx` with this code.

Features:
1. Displays current language (default "English").
2. Clicking opens a dropdown menu.
3. Selecting "Urdu" triggers the translation API.
4. Selecting "English" reloads the page (to revert translation).

Code:
```tsx
import React, { useState } from 'react';

export default function TranslateButton() {
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [currentLang, setCurrentLang] = useState('English');

  const handleTranslate = async () => {
    setIsOpen(false);
    if (loading || currentLang === 'Urdu') return;

    setLoading(true);
    setCurrentLang('Urdu');

    // 1. Find content
    const contentDiv = document.querySelector('.markdown') || document.querySelector('article');
    if (!contentDiv) {
      setLoading(false);
      return;
    }

    const originalText = contentDiv.innerText;
    const textToSend = originalText.substring(0, 1500) + "... (Truncated)";

    try {
      const res = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToSend })
      });

      const data = await res.json();
      if (data.translated_text) {
        contentDiv.innerHTML = `<div dir="rtl" style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.2rem; line-height: 2; padding: 20px;">${data.translated_text}</div>`;
      }
    } catch (e) {
      alert("Backend not connected!");
      setCurrentLang('English'); // Revert on error
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (currentLang === 'Urdu') {
        window.location.reload(); // Simplest way to restore original DOM
    }
    setIsOpen(false);
  };

  return (
    <div className="navbar__item" style={{ position: 'relative' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="clean-btn navbar__link"
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontWeight: 500 }}
      >
        {/* Language Icon */}
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
             <path d="M5 8l6 11"></path>
             <path d="M11 8l6 11"></path>
             <path d="M4 19h16"></path>
             <path d="M12 2L12 2"></path>
             <text x="4" y="16" fontSize="14" fontFamily="Arial" stroke="none" fill="currentColor">文</text>
        </svg>

        {/* Current Language Text */}
        <span>{currentLang}</span>

        {/* Loading Spinner or Chevron */}
        {loading ? (
           <span>⏳</span>
        ) : (
           <span style={{ fontSize: '0.8em', transform: isOpen ? 'rotate(180deg)' : 'rotate(0)' }}>▼</span>
        )}
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <ul style={{
          position: 'absolute', top: '100%', right: 0,
          background: 'var(--ifm-navbar-background-color)',
          border: '1px solid var(--ifm-color-emphasis-200)',
          borderRadius: '8px', boxShadow: '0 4px 10px rgba(0,0,0,0.1)',
          listStyle: 'none', padding: '5px 0', margin: 0, minWidth: '120px',
          zIndex: 100
        }}>
          <li style={{ padding: '0' }}>
            <button
              onClick={handleReset}
              style={{
                display: 'block', width: '100%', textAlign: 'left', padding: '8px 15px',
                background: 'transparent', border: 'none', cursor: 'pointer',
                color: currentLang === 'English' ? 'var(--ifm-color-primary)' : 'inherit',
                fontWeight: currentLang === 'English' ? 'bold' : 'normal'
              }}
            >
              English
            </button>
          </li>
          <li style={{ padding: '0' }}>
            <button
              onClick={handleTranslate}
              style={{
                display: 'block', width: '100%', textAlign: 'left', padding: '8px 15px',
                background: 'transparent', border: 'none', cursor: 'pointer',
                color: currentLang === 'Urdu' ? 'var(--ifm-color-primary)' : 'inherit',
                fontWeight: currentLang === 'Urdu' ? 'bold' : 'normal'
              }}
            >
              Urdu (اردو)
            </button>
          </li>
        </ul>
      )}
    </div>
  );
}
```'
labels: ["react", "docusaurus", "translation", "dropdown", "language-selection"]
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

Task: Upgrade Translate Button to Dropdown Menu

Action: Rewrite `frontend/src/components/TranslateButton.tsx` with this code.

Features:
1. Displays current language (default "English").
2. Clicking opens a dropdown menu.
3. Selecting "Urdu" triggers the translation API.
4. Selecting "English" reloads the page (to revert translation).

Code:
```tsx
import React, { useState } from 'react';

export default function TranslateButton() {
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [currentLang, setCurrentLang] = useState('English');

  const handleTranslate = async () => {
    setIsOpen(false);
    if (loading || currentLang === 'Urdu') return;

    setLoading(true);
    setCurrentLang('Urdu');

    // 1. Find content
    const contentDiv = document.querySelector('.markdown') || document.querySelector('article');
    if (!contentDiv) {
      setLoading(false);
      return;
    }

    const originalText = contentDiv.innerText;
    const textToSend = originalText.substring(0, 1500) + "... (Truncated)";

    try {
      const res = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToSend })
      });

      const data = await res.json();
      if (data.translated_text) {
        contentDiv.innerHTML = `<div dir="rtl" style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.2rem; line-height: 2; padding: 20px;">${data.translated_text}</div>`;
      }
    } catch (e) {
      alert("Backend not connected!");
      setCurrentLang('English'); // Revert on error
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (currentLang === 'Urdu') {
        window.location.reload(); // Simplest way to restore original DOM
    }
    setIsOpen(false);
  };

  return (
    <div className="navbar__item" style={{ position: 'relative' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="clean-btn navbar__link"
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontWeight: 500 }}
      >
        {/* Language Icon */}
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
             <path d="M5 8l6 11"></path>
             <path d="M11 8l6 11"></path>
             <path d="M4 19h16"></path>
             <path d="M12 2L12 2"></path>
             <text x="4" y="16" fontSize="14" fontFamily="Arial" stroke="none" fill="currentColor">文</text>
        </svg>

        {/* Current Language Text */}
        <span>{currentLang}</span>

        {/* Loading Spinner or Chevron */}
        {loading ? (
           <span>⏳</span>
        ) : (
           <span style={{ fontSize: '0.8em', transform: isOpen ? 'rotate(180deg)' : 'rotate(0)' }}>▼</span>
        )}
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <ul style={{
          position: 'absolute', top: '100%', right: 0,
          background: 'var(--ifm-navbar-background-color)',
          border: '1px solid var(--ifm-color-emphasis-200)',
          borderRadius: '8px', boxShadow: '0 4px 10px rgba(0,0,0,0.1)',
          listStyle: 'none', padding: '5px 0', margin: 0, minWidth: '120px',
          zIndex: 100
        }}>
          <li style={{ padding: '0' }}>
            <button
              onClick={handleReset}
              style={{
                display: 'block', width: '100%', textAlign: 'left', padding: '8px 15px',
                background: 'transparent', border: 'none', cursor: 'pointer',
                color: currentLang === 'English' ? 'var(--ifm-color-primary)' : 'inherit',
                fontWeight: currentLang === 'English' ? 'bold' : 'normal'
              }}
            >
              English
            </button>
          </li>
          <li style={{ padding: '0' }}>
            <button
              onClick={handleTranslate}
              style={{
                display: 'block', width: '100%', textAlign: 'left', padding: '8px 15px',
                background: 'transparent', border: 'none', cursor: 'pointer',
                color: currentLang === 'Urdu' ? 'var(--ifm-color-primary)' : 'inherit',
                fontWeight: currentLang === 'Urdu' ? 'bold' : 'normal'
              }}
            >
              Urdu (اردو)
            </button>
          </li>
        </ul>
      )}
    </div>
  );
}
```

## Response snapshot

I have updated `frontend/src/components/TranslateButton.tsx` to include the dropdown menu functionality. Now, the button will display the current language, and clicking it will reveal a dropdown to switch between English and Urdu. Selecting Urdu will trigger the translation, and selecting English will reload the page to revert the translation.

## Outcome

- ✅ Impact: Upgraded the Translate Button to a dropdown menu with language selection.
- 🧪 Tests: N/A (Requires manual verification by the user).
- 📁 Files: `frontend/src/components/TranslateButton.tsx` modified.
- 🔁 Next prompts: Awaiting user confirmation of the new dropdown functionality and translation.
- 🧠 Reflection: Successfully implemented the requested upgrade to a dropdown menu for language selection, enhancing the user experience for translation. Provided clear instructions for the user to verify the changes.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

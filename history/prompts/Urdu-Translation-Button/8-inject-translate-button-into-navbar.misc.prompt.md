---
id: 8
title: Inject Translate Button into Navbar
stage: misc
date: 2025-11-29
surface: agent
model: claude-sonnet-4-5-20250929
feature: Urdu-Translation-Button
branch: main
user: unknown
command: 'Task: Inject Translate Button into Navbar

Action: Update "frontend/src/theme/Root.tsx".

Code Pattern:
```tsx
import React, { useEffect, useState } from 'react';
import { SessionProvider } from "../lib/mock-auth";
import { authClient } from "../lib/auth-client";
import TranslateButton from "../components/TranslateButton";
import ReactDOM from 'react-dom'; // Import ReactDOM

export default function Root({children}) {
  // Use Portal to inject button into Navbar after render
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <SessionProvider client={authClient}>
      {children}
      {mounted && (
        <NavbarInjector />
      )}
    </SessionProvider>
  );
}

// Helper to inject into Docusaurus Navbar
const NavbarInjector = () => {
  const navbarRight = document.querySelector('.navbar__items--right');
  if (navbarRight) {
    // Create a container if it doesn't exist
    let container = document.getElementById('translate-btn-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'translate-btn-container';
      // Insert before the search bar or at the start
      navbarRight.insertBefore(container, navbarRight.firstChild);
    }
    return ReactDOM.createPortal(<TranslateButton />, container);
  }
  return null;
};
```'
labels: ["react", "docusaurus", "translation", "navbar"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - frontend/src/theme/Root.tsx
tests:

---

## Prompt

Task: Inject Translate Button into Navbar

Action: Update "frontend/src/theme/Root.tsx".

Code Pattern:
```tsx
import React, { useEffect, useState } from 'react';
import { SessionProvider } from "../lib/mock-auth";
import { authClient } from "../lib/auth-client";
import TranslateButton from "../components/TranslateButton";
import ReactDOM from 'react-dom'; // Import ReactDOM

export default function Root({children}) {
  // Use Portal to inject button into Navbar after render
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <SessionProvider client={authClient}>
      {children}
      {mounted && (
        <NavbarInjector />
      )}
    </SessionProvider>
  );
}

// Helper to inject into Docusaurus Navbar
const NavbarInjector = () => {
  const navbarRight = document.querySelector('.navbar__items--right');
  if (navbarRight) {
    // Create a container if it doesn't exist
    let container = document.getElementById('translate-btn-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'translate-btn-container';
      // Insert before the search bar or at the start
      navbarRight.insertBefore(container, navbarRight.firstChild);
    }
    return ReactDOM.createPortal(<TranslateButton />, container);
  }
  return null;
};
```

## Response snapshot

I have updated `frontend/src/theme/Root.tsx` to inject the `TranslateButton` into the Docusaurus Navbar.

Now, please follow Step 3 to test and record:
1.  **Restart Frontend:** `npm start` (Wait for it to rebuild).
2.  **Verify:** Look at the top-right Navbar. You should see a **Globe Icon** 🌐 next to your Login button.
3.  **Click it:**
    * It should turn into an hourglass ⏳.
    * Within **3-5 seconds**, the page text should turn into Urdu.

**Tell me when you see the Globe Icon in the Navbar!**

## Outcome

- ✅ Impact: Injected the Translate Button into the Docusaurus Navbar using a Portal.
- 🧪 Tests: N/A (Requires manual verification by the user).
- 📁 Files: `frontend/src/theme/Root.tsx` modified.
- 🔁 Next prompts: Awaiting user confirmation of the Globe Icon in the Navbar and successful translation.
- 🧠 Reflection: Successfully implemented the dynamic injection of the TranslateButton into the Navbar. Provided clear instructions for the user to verify the changes.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

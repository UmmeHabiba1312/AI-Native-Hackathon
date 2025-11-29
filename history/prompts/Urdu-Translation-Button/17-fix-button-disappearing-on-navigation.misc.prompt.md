---
id: 17
title: Fix Translate Button Disappearing on Navigation
stage: misc
date: 2025-11-29
surface: agent
model: claude-sonnet-4-5-20250929
feature: Urdu-Translation-Button
branch: main
user: unknown
command: 'Task: Fix Button Disappearing on Navigation.

Action: Update `frontend/src/theme/Root.tsx` with this robust version.

Changes:
1. Import `useLocation` from `@docusaurus/router`.
2. Add `key={location.pathname}` to the Injector. This forces React to "reset" the button logic every time the user navigates, ensuring it re-attaches to the DOM.

Code:
```tsx
import React, { useEffect, useState } from 'react';
import { useLocation } from '@docusaurus/router'; // Import location hook
import { SessionProvider } from "../lib/mock-auth";
import authClient from "../lib/auth-client";
import TranslateButton from "../components/TranslateButton";
import ChatWidget from "../components/ChatWidget";
import ReactDOM from 'react-dom';

export default function Root({children}) {
  const [mounted, setMounted] = useState(false);
  const location = useLocation(); // Track current page

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <SessionProvider client={authClient}>
      {children}

      {/* Chat Widget stays forever */}
      <ChatWidget />

      {/* Force re-inject when location changes using 'key' */}
      {mounted && (
        <NavbarInjector key={location.pathname} />
      )}
    </SessionProvider>
  );
}

// Self-contained component to handle DOM injection
const NavbarInjector = () => {
  const [container, setContainer] = useState<HTMLElement | null>(null);

  useEffect(() => {
    // Attempt to find or create the container
    const prepareContainer = () => {
      const navbarRight = document.querySelector('.navbar__items--right');
      if (navbarRight) {
        let el = document.getElementById('translate-btn-container');
        if (!el) {
          el = document.createElement('div');
          el.id = 'translate-btn-container';
          // Insert as first item in right navbar
          navbarRight.insertBefore(el, navbarRight.firstChild);
        }
        setContainer(el);
      }
    };

    // Run immediately
    prepareContainer();

    // Retry strictly for safety (in case Navbar renders slowly)
    const timer = setTimeout(prepareContainer, 50);
    return () => clearTimeout(timer);
  }, []);

  if (!container) return null;
  return ReactDOM.createPortal(<TranslateButton />, container);
};
```'
labels: ["react", "docusaurus", "navbar", "bug-fix"]
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

Task: Fix Button Disappearing on Navigation.

Action: Update `frontend/src/theme/Root.tsx` with this robust version.

Changes:
1. Import `useLocation` from `@docusaurus/router`.
2. Add `key={location.pathname}` to the Injector. This forces React to "reset" the button logic every time the user navigates, ensuring it re-attaches to the DOM.

Code:
```tsx
import React, { useEffect, useState } from 'react';
import { useLocation } from '@docusaurus/router'; // Import location hook
import { SessionProvider } from "../lib/mock-auth";
import authClient from "../lib/auth-client";
import TranslateButton from "../components/TranslateButton";
import ChatWidget from "../components/ChatWidget";
import ReactDOM from 'react-dom';

export default function Root({children}) {
  const [mounted, setMounted] = useState(false);
  const location = useLocation(); // Track current page

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <SessionProvider client={authClient}>
      {children}

      {/* Chat Widget stays forever */}
      <ChatWidget />

      {/* Force re-inject when location changes using 'key' */}
      {mounted && (
        <NavbarInjector key={location.pathname} />
      )}
    </SessionProvider>
  );
}

// Self-contained component to handle DOM injection
const NavbarInjector = () => {
  const [container, setContainer] = useState<HTMLElement | null>(null);

  useEffect(() => {
    // Attempt to find or create the container
    const prepareContainer = () => {
      const navbarRight = document.querySelector('.navbar__items--right');
      if (navbarRight) {
        let el = document.getElementById('translate-btn-container');
        if (!el) {
          el = document.createElement('div');
          el.id = 'translate-btn-container';
          // Insert as first item in right navbar
          navbarRight.insertBefore(el, navbarRight.firstChild);
        }
        setContainer(el);
      }
    };

    // Run immediately
    prepareContainer();

    // Retry strictly for safety (in case Navbar renders slowly)
    const timer = setTimeout(prepareContainer, 50);
    return () => clearTimeout(timer);
  }, []);

  if (!container) return null;
  return ReactDOM.createPortal(<TranslateButton />, container);
};
```

## Response snapshot

I have updated `frontend/src/theme/Root.tsx` with the more robust version to prevent the translate button from disappearing on navigation. This includes importing `useLocation` and using `key={location.pathname}` on the `NavbarInjector` to force its re-rendering.

## Outcome

- ✅ Impact: Implemented a fix to ensure the translate button persists across page navigations.
- 🧪 Tests: N/A (Requires manual verification by the user).
- 📁 Files: `frontend/src/theme/Root.tsx` modified.
- 🔁 Next prompts: Awaiting user confirmation of the fix.
- 🧠 Reflection: Successfully addressed the navigation issue by introducing a key prop to force re-mounting of the NavbarInjector, ensuring consistent visibility of the translate button.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

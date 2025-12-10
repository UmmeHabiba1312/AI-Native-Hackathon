import React, { useEffect, useState } from 'react';
import { useLocation } from '@docusaurus/router'; // Import location hook
import { SessionProvider } from "../lib/mock-auth";
import authClient from "../lib/auth-client";
import TranslateButton from "../components/TranslateButton";
import PersonalizeButton from "../components/PersonalizeButton";
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
  const [translateContainer, setTranslateContainer] = useState<HTMLElement | null>(null);
  const [personalizeContainer, setPersonalizeContainer] = useState<HTMLElement | null>(null);

  useEffect(() => {
    const prepareContainers = () => {
      const navbarRight = document.querySelector('.navbar__items--right');
      if (navbarRight) {
        // Translate Button Container
        let translateEl = document.getElementById('translate-btn-container');
        if (!translateEl) {
          translateEl = document.createElement('div');
          translateEl.id = 'translate-btn-container';
          navbarRight.insertBefore(translateEl, navbarRight.firstChild);
        }
        setTranslateContainer(translateEl);

        // Personalize Button Container
        let personalizeEl = document.getElementById('personalize-btn-container');
        if (!personalizeEl) {
          personalizeEl = document.createElement('div');
          personalizeEl.id = 'personalize-btn-container';
          // Insert after translate button container, or as first if translate is not there
          navbarRight.insertBefore(personalizeEl, translateEl ? translateEl.nextSibling : navbarRight.firstChild);
        }
        setPersonalizeContainer(personalizeEl);
      }
    };

    prepareContainers();
    const timer = setTimeout(prepareContainers, 50);
    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      {translateContainer && ReactDOM.createPortal(<TranslateButton />, translateContainer)}
      {personalizeContainer && ReactDOM.createPortal(<PersonalizeButton />, personalizeContainer)}
    </>
  );
};
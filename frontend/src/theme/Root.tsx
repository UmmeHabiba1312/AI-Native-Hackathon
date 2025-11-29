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
import React, { useState, useEffect } from 'react';

export default function TranslateButton() {
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [currentLang, setCurrentLang] = useState('English');

  // 1. Auto-Trigger on Load if Urdu was saved
  useEffect(() => {
    const savedLang = sessionStorage.getItem('hackathon_lang');
    if (savedLang === 'Urdu') {
      setCurrentLang('Urdu');
      // Delay slightly to allow DOM to render before translating
      setTimeout(() => handleTranslate(true), 800); 
    }
  }, []);

  const handleTranslate = async (isAuto = false) => {
    setIsOpen(false);
    // Prevent double-firing if already loading (unless it's the auto-trigger)
    if (loading && !isAuto) return;
    
    setLoading(true);
    setCurrentLang('Urdu');
    sessionStorage.setItem('hackathon_lang', 'Urdu'); // Save to memory

    // Find content
    const contentDiv = (document.querySelector('.markdown') ||
                        document.querySelector('article') ||
                        document.querySelector('main')) as HTMLElement;
    
    if (!contentDiv) {
      console.log("Translation Error: Content div not found");
      setLoading(false);
      return;
    }

    const originalText = contentDiv.innerText;
    // Truncate for speed
    const textToSend = originalText.substring(0, 1500) + "...";

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
      console.error(e);
      if (!isAuto) alert("Backend not connected!");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    sessionStorage.removeItem('hackathon_lang'); // Clear memory
    window.location.reload(); // Reload to restore English
  };

  return (
    <div className="navbar__item" style={{ position: 'relative' }}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="clean-btn navbar__link"
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontWeight: 500 }}
      >
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
             <path d="M5 8l6 11"></path>
             <path d="M11 8l6 11"></path>
             <path d="M4 19h16"></path>
             <path d="M12 2L12 2"></path>
             <text x="4" y="16" fontSize="14" fontFamily="Arial" stroke="none" fill="currentColor">文</text>
        </svg>
        <span>{currentLang}</span>
        {loading ? <span>⏳</span> : <span style={{ fontSize: '0.8em', transform: isOpen ? 'rotate(180deg)' : 'rotate(0)' }}>▼</span>}
      </button>

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
            <button onClick={handleReset} style={{ display: 'block', width: '100%', textAlign: 'left', padding: '8px 15px', background: 'transparent', border: 'none', cursor: 'pointer', color: 'inherit' }}>English</button>
          </li>
          <li style={{ padding: '0' }}>
            <button onClick={() => handleTranslate(false)} style={{ display: 'block', width: '100%', textAlign: 'left', padding: '8px 15px', background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--ifm-color-primary)', fontWeight: 'bold' }}>Urdu (اردو)</button>
          </li>
        </ul>
      )}
    </div>
  );
}
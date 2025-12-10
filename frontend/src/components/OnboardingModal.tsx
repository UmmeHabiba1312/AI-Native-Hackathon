import React, { useState, useEffect } from 'react';
import { useSession } from "../lib/mock-auth";

const OnboardingModal = () => {
  // Destructure 'data' from the hook, rename it to 'session' for clarity
  const { data: session } = useSession();

  const [isOpen, setIsOpen] = useState(false);
  const [pythonExperience, setPythonExperience] = useState('Beginner'); // Default to Beginner

  useEffect(() => {
    // Check if logged in AND if onboarding is incomplete in localStorage
    if (session?.user) {
      const userContext = JSON.parse(localStorage.getItem('user_context') || '{}');
      // Open if hasGPU or level are not set in localStorage
      if (userContext.hasGPU === undefined || userContext.level === undefined) {
        setIsOpen(true);
      }
    }
  }, [session]);

  if (!isOpen) return null;

  const handleSavePreferences = (hasGPU: boolean) => {
    const userContext = {
      hasGPU,
      level: pythonExperience,
    };
    localStorage.setItem('user_context', JSON.stringify(userContext));
    setIsOpen(false); // Close modal after saving
  };

  return (
    <div className="onboarding-modal-overlay" style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.7)', zIndex: 9999,
      display: 'flex', justifyContent: 'center', alignItems: 'center'
    }}>
      <div className="onboarding-modal" style={{
        backgroundColor: 'white', padding: '2rem', borderRadius: '10px',
        maxWidth: '400px', width: '90%', textAlign: 'center', color: 'black'
      }}>
        <h2>🚀 Hackathon Setup</h2>
        <p>To personalize your experience, please answer:</p>

        <div style={{ margin: '20px 0', textAlign: 'left' }}>
          <label style={{ display: 'block', marginBottom: '10px', fontWeight: 'bold' }}>
            Do you have an NVIDIA GPU?
          </label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={() => handleSavePreferences(true)} style={{ flex: 1, padding: '8px', cursor: 'pointer' }}>Yes (RTX 4070+)</button>
            <button onClick={() => handleSavePreferences(false)} style={{ flex: 1, padding: '8px', cursor: 'pointer' }}>No (Cloud Only)</button>
          </div>
        </div>

        <div style={{ margin: '20px 0', textAlign: 'left' }}>
          <label style={{ display: 'block', marginBottom: '10px', fontWeight: 'bold' }}>
            Python Experience?
          </label>
          <select
            style={{ width: '100%', padding: '8px' }}
            value={pythonExperience}
            onChange={(e) => setPythonExperience(e.target.value)}
          >
            <option>Beginner</option>
            <option>Intermediate</option>
            <option>Advanced</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default OnboardingModal;
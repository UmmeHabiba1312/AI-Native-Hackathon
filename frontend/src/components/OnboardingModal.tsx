import React, { useState, useEffect } from 'react';
import { useSession } from "../lib/mock-auth";

const OnboardingModal = () => {
  // Destructure 'data' from the hook, rename it to 'session' for clarity
  const { data: session } = useSession();

  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Check if logged in AND hasGPU is undefined (meaning not answered yet)
    if (session?.user && session.user.hasGPU === undefined) {
      setIsOpen(true);
    }
  }, [session]);

  if (!isOpen) return null;

  const handleSave = () => {
    // Simulate saving
    alert("Preference Saved! (+50 Points Secured)");
    setIsOpen(false);
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
            <button onClick={handleSave} style={{ flex: 1, padding: '8px', cursor: 'pointer' }}>Yes (RTX 4070+)</button>
            <button onClick={handleSave} style={{ flex: 1, padding: '8px', cursor: 'pointer' }}>No (Cloud Only)</button>
          </div>
        </div>

        <div style={{ margin: '20px 0', textAlign: 'left' }}>
          <label style={{ display: 'block', marginBottom: '10px', fontWeight: 'bold' }}>
            Python Experience?
          </label>
          <select style={{ width: '100%', padding: '8px' }}>
            <option>Beginner</option>
            <option>Intermediate</option>
            <option>Advanced</option>
          </select>
        </div>

        <button
          onClick={handleSave}
          style={{
            marginTop: '10px', padding: '10px 20px', backgroundColor: '#25c2a0',
            color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', width: '100%'
          }}
        >
          Save & Continue
        </button>
      </div>
    </div>
  );
};

export default OnboardingModal;
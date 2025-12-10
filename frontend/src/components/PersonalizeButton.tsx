import React from 'react';

const PersonalizeButton = () => {
  const handlePersonalize = async () => {
    const userContextString = localStorage.getItem('user_context');
    if (!userContextString) {
      alert('Please complete the survey first!');
      return;
    }

    try {
      const userContext = JSON.parse(userContextString);
      const pageContent = document.querySelector('main')?.innerText || '';

      const response = await fetch('https://ai-native-hackathon-backend.vercel.app/personalize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: pageContent,
          context: userContext,
        }),
      });

      const data = await response.json();

      if (data.personalized_text) {
        const mainElement = document.querySelector('main');
        if (mainElement) {
          mainElement.innerText = data.personalized_text;
        }
      } else {
        alert(data.personalized_text || 'Personalization failed.');
      }
    } catch (error) {
      console.error('Personalization error:', error);
      alert('An error occurred during personalization.');
    }
  };

  return (
    <button
      onClick={handlePersonalize}
      style={{
        padding: '8px 16px',
        backgroundColor: '#007bff',
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
        marginLeft: '10px',
      }}
    >
      Personalize
    </button>
  );
};

export default PersonalizeButton;
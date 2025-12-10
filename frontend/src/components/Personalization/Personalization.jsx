import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './Personalization.module.css';

// Placeholder component for personalization features
const Personalization = () => {
  const [userProfile, setUserProfile] = useState({
    background: 'software',
    preferences: {
      language: 'en',
      difficulty_level: 'intermediate'
    },
    progress: {}
  });
  const [isEditing, setIsEditing] = useState(false);

  // In a real implementation, this would fetch user profile from the API
  useEffect(() => {
    // const fetchUserProfile = async () => {
    //   // API call to get user profile
    //   const response = await fetch('/api/v1/users/profile');
    //   const data = await response.json();
    //   setUserProfile(data.profile);
    // };
    // fetchUserProfile();
  }, []);

  const handleSaveProfile = async () => {
    // In a real implementation, this would save to the backend
    // const response = await fetch('/api/v1/users/profile', {
    //   method: 'PUT',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ profile: userProfile })
    // });

    setIsEditing(false);
    alert('Profile updated successfully!');
  };

  return (
    <div className={styles.personalization}>
      <div className={styles.header}>
        <h3>Personalization Settings</h3>
        <button
          className={styles.editButton}
          onClick={() => setIsEditing(!isEditing)}
        >
          {isEditing ? 'Cancel' : 'Edit Profile'}
        </button>
      </div>

      {isEditing ? (
        <div className={styles.editForm}>
          <div className={styles.formGroup}>
            <label htmlFor="background">Background:</label>
            <select
              id="background"
              value={userProfile.background}
              onChange={(e) => setUserProfile({
                ...userProfile,
                background: e.target.value
              })}
              className={styles.select}
            >
              <option value="software">Software Development</option>
              <option value="hardware">Hardware Engineering</option>
              <option value="robotics">Robotics</option>
              <option value="ai">AI/Machine Learning</option>
              <option value="beginner">Complete Beginner</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="difficulty">Preferred Difficulty:</label>
            <select
              id="difficulty"
              value={userProfile.preferences.difficulty_level}
              onChange={(e) => setUserProfile({
                ...userProfile,
                preferences: {
                  ...userProfile.preferences,
                  difficulty_level: e.target.value
                }
              })}
              className={styles.select}
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="language">Preferred Language:</label>
            <select
              id="language"
              value={userProfile.preferences.language}
              onChange={(e) => setUserProfile({
                ...userProfile,
                preferences: {
                  ...userProfile.preferences,
                  language: e.target.value
                }
              })}
              className={styles.select}
            >
              <option value="en">English</option>
              <option value="ur">Urdu</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
            </select>
          </div>

          <button
            className={styles.saveButton}
            onClick={handleSaveProfile}
          >
            Save Profile
          </button>
        </div>
      ) : (
        <div className={styles.profileDisplay}>
          <div className={styles.profileItem}>
            <strong>Background:</strong> {userProfile.background}
          </div>
          <div className={styles.profileItem}>
            <strong>Difficulty Level:</strong> {userProfile.preferences.difficulty_level}
          </div>
          <div className={styles.profileItem}>
            <strong>Language:</strong> {userProfile.preferences.language}
          </div>
        </div>
      )}

      <div className={styles.personalizationInfo}>
        <h4>How Personalization Works</h4>
        <p>
          Your profile helps us tailor the content to your background and learning preferences.
          Content difficulty and explanations will be adjusted based on your selected preferences.
        </p>
      </div>
    </div>
  );
};

export default Personalization;
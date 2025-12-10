import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './Translation.module.css';

// Placeholder component for translation features
const Translation = ({ content, onContentChange }) => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [translatedContent, setTranslatedContent] = useState(null);

  const supportedLanguages = [
    { code: 'en', name: 'English' },
    { code: 'ur', name: 'Urdu' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' }
  ];

  const handleTranslate = async (targetLanguage) => {
    if (targetLanguage === currentLanguage) {
      // Switch back to original language
      setTranslatedContent(null);
      setCurrentLanguage('en');
      if (onContentChange) {
        onContentChange(content);
      }
      return;
    }

    setIsTranslating(true);

    try {
      // In a real implementation, this would call the translation API
      // const response = await fetch('/api/v1/translate', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({
      //     text: content,
      //     target_language: targetLanguage,
      //     source_language: currentLanguage
      //   })
      // });
      // const data = await response.json();

      // For now, simulate translation with a placeholder
      await new Promise(resolve => setTimeout(resolve, 800));

      // Placeholder translation - in real implementation, this would come from the API
      const placeholderTranslation = `[TRANSLATED TO ${targetLanguage.toUpperCase()}] ${content.substring(0, 100)}...`;

      setTranslatedContent(placeholderTranslation);
      setCurrentLanguage(targetLanguage);

      if (onContentChange) {
        onContentChange(placeholderTranslation);
      }
    } catch (error) {
      console.error('Translation error:', error);
      alert('Error translating content. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <div className={styles.translation}>
      <div className={styles.translationHeader}>
        <h3>Translation</h3>
        <div className={styles.languageSelector}>
          {supportedLanguages.map((lang) => (
            <button
              key={lang.code}
              className={clsx(
                styles.languageButton,
                currentLanguage === lang.code && styles.active
              )}
              onClick={() => handleTranslate(lang.code)}
              disabled={isTranslating}
            >
              {lang.name}
              {currentLanguage === lang.code && ' ✓'}
            </button>
          ))}
        </div>
      </div>

      {isTranslating && (
        <div className={styles.translatingIndicator}>
          Translating... <span className={styles.spinner}></span>
        </div>
      )}

      <div className={styles.translationInfo}>
        <p>
          {currentLanguage === 'en'
            ? 'Content is displayed in English. Select another language to translate.'
            : `Content is currently displayed in ${supportedLanguages.find(l => l.code === currentLanguage)?.name}.`}
        </p>
      </div>

      {currentLanguage !== 'en' && (
        <div className={styles.translationNote}>
          <small>
            Note: This is a placeholder for translation functionality.
            In the full implementation, content would be translated using AI models.
          </small>
        </div>
      )}
    </div>
  );
};

export default Translation;
from pydantic import BaseModel


class TranslationRequest(BaseModel):
    text: str
    target_language: str  # e.g., ur, es, fr
    source_language: str = "en"  # default source language


class TranslationResponse(BaseModel):
    translated_text: str
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Hashmi2004/multilingual-toxic-comment-xlm-roberta"

@st.cache_resource(show_spinner="Downloading & Loading Model (XLM-R)...")
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception as e:
        return None, None

def predict_comment(text: str) -> dict:
    """
    Predicts if a given text is toxic or non-toxic using XLM-R.
    """
    try:
        tokenizer, model = load_model()
        if tokenizer is None or model is None:
            return {
                "status": "error",
                "message": "Could not load the model from Hugging Face."
            }
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            class_id = predicted_class.item()
            conf_score = confidence.item() * 100
            
            label = "Toxic" if class_id == 1 else "Non-toxic"
            
            return {
                "status": "success",
                "prediction": label,
                "confidence": conf_score
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

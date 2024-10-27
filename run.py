# run.py

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from dotenv import load_dotenv

# โหลด environment variables
load_dotenv('config/.env')

# Import local modules
from src.model import ChatbotModel, ResponseGenerator
from src.preprocess import TextPreprocessor
from src.utils import ChatbotUtils

# Initialize FastAPI
app = FastAPI(
    title="iPhone Store Chatbot API",
    description="API สำหรับ Chatbot ร้านขาย iPhone",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    product_info: Optional[Dict[str, Any]] = None

# Initialize components
preprocessor = None
chatbot_model = None
response_generator = None
API_KEY = os.getenv('API_KEY')

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.on_event("startup")
async def load_model():
    global preprocessor, chatbot_model, response_generator
    
    try:
        # Initialize preprocessor
        preprocessor = TextPreprocessor(
            max_words=int(os.getenv('MAX_WORDS', 2000)),
            max_len=int(os.getenv('MAX_LENGTH', 50))
        )
        
        # Load training data to setup tokenizer and label encoder
        texts, intents = preprocessor.load_training_data('data/training_data.csv')
        preprocessor.prepare_data(texts, intents)
        
        # Load responses
        responses = preprocessor.load_responses('data/responses.json')
        
        # Initialize and load model
        chatbot_model = ChatbotModel(
            vocab_size=preprocessor.get_vocab_size(),
            num_classes=preprocessor.get_num_classes(),
            max_len=int(os.getenv('MAX_LENGTH', 50))
        )
        chatbot_model.load_model('models/chatbot_model.h5')
        
        # Initialize response generator
        response_generator = ResponseGenerator(responses)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.get("/")
async def root():
    return {"message": "iPhone Store Chatbot API"}

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    try:
        # Prepare input
        X = preprocessor.prepare_data([request.message], train=False)
        
        # Get predictions and confidence
        predictions = chatbot_model.model.predict(X)
        intent_idx = predictions.argmax(axis=1)[0]
        confidence = float(predictions.max(axis=1)[0])
        
        # Get intent name
        intent = preprocessor.decode_intent(intent_idx)
        
        # Extract product info
        product_model, storage = ChatbotUtils.extract_product_info(request.message)
        
        # Generate response
        response = response_generator.get_response(
            intent,
            model=product_model,
            storage=storage
        )
        
        # Save chat history
        if request.chat_id:
            ChatbotUtils.save_chat_history(
                chat_id=request.chat_id,
                message=request.message,
                response=response,
                intent=intent
            )
        
        return ChatResponse(
            response=response,
            intent=intent,
            confidence=confidence,
            product_info={
                "model": product_model,
                "storage": storage
            } if product_model or storage else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/intents", dependencies=[Depends(verify_api_key)])
async def get_intents():
    """Get all available intents"""
    try:
        return {"intents": preprocessor.label_encoder.classes_.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products", dependencies=[Depends(verify_api_key)])
async def get_products():
    """Get all iPhone models and their details"""
    try:
        responses = preprocessor.load_responses('data/responses.json')
        return {"products": responses.get("price_inquiry", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 8000)),
        reload=bool(os.getenv('DEBUG', 'True').lower() == 'true')
    )
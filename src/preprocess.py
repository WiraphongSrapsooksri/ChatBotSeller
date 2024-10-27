# preprocess.py

import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder
import json
import re

class TextPreprocessor:
    def __init__(self, max_words=2000, max_len=50):  # เปลี่ยนค่าเริ่มต้นให้ตรงกัน
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text):
        """ปรับปรุงการทำความสะอาดข้อความ"""
        # ลบช่องว่างที่ไม่จำเป็น
        text = re.sub(r'\s+', ' ', str(text).strip())
        
        # แทนที่คำที่พบบ่อยด้วยรูปแบบมาตรฐาน
        text = text.lower()
        text = text.replace('iphone15', 'iphone 15')
        text = text.replace('promax', 'pro max')
        text = text.replace('ราคาเท่าไหร่', 'ราคาเท่าไร')
        text = text.replace('ราคากี่บาท', 'ราคาเท่าไร')
        text = text.replace('ขายเท่าไร', 'ราคาเท่าไร')
        
        # normalize ข้อความภาษาไทย
        text = normalize(text)
        
        return text
    
    def tokenize_text(self, text):
        """แบ่งคำภาษาไทย"""
        return word_tokenize(text, engine='newmm')
    
    def prepare_data(self, texts, intents=None, train=True):
        """ปรับปรุงการเตรียมข้อมูล"""
        # ทำความสะอาดข้อความ
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        if train:
            # fit tokenizer กับข้อความทั้งหมด
            self.tokenizer.fit_on_texts(cleaned_texts)
        
        # แปลงข้อความเป็นลำดับตัวเลข
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        if intents is not None and train:
            self.label_encoder.fit_transform(intents)
        
        if intents is not None:
            y = self.label_encoder.transform(intents)
            return X, y
        
        return X
    
    def load_training_data(self, csv_path):
        """โหลดข้อมูลสำหรับเทรนจาก CSV"""
        df = pd.read_csv(csv_path)
        return df['text'].values, df['intent'].values
    
    def load_responses(self, json_path):
        """โหลดข้อมูลการตอบกลับจาก JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        return responses
    
    def get_vocab_size(self):
        """ขนาดของ vocabulary"""
        return min(len(self.tokenizer.word_index) + 1, self.max_words)
    
    def get_num_classes(self):
        """จำนวน intent ทั้งหมด"""
        return len(self.label_encoder.classes_)
    
    def decode_intent(self, intent_idx):
        """แปลงเลข intent กลับเป็นชื่อ intent"""
        return self.label_encoder.inverse_transform([intent_idx])[0]
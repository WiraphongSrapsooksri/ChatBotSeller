# train_model.py

import numpy as np
import pandas as pd
from src.model import ChatbotModel
from src.preprocess import TextPreprocessor
from src.utils import ChatbotUtils
from training_visualizer import TrainingVisualizer
from pathlib import Path
from dotenv import load_dotenv
import os
from collections import Counter

def train():
    try:
        print("เริ่มกระบวนการเทรนโมเดล...")
        
        # สร้างโฟลเดอร์เก็บโมเดลและกราฟ
        Path('models').mkdir(exist_ok=True)
        Path('plots').mkdir(exist_ok=True)
        
        # โหลดและเตรียมข้อมูล
        preprocessor = TextPreprocessor(
            max_words=int(os.getenv('MAX_WORDS', 2000)),
            max_len=int(os.getenv('MAX_LENGTH', 50))
        )
        
        print("กำลังโหลดข้อมูล...")
        # โหลดข้อมูลเป็น DataFrame
        df = pd.read_csv('data/training_data.csv')
        
        # ลบแถวที่มี NaN
        df = df.dropna()
        
        texts = df['text'].values
        intents = df['intent'].values
        
        print(f"จำนวนข้อมูลทั้งหมด: {len(texts)}")
        print(f"จำนวน intents: {len(set(intents))}")
        
        print("\nการกระจายของข้อมูล:")
        intent_counts = Counter(intents)
        for intent, count in intent_counts.items():
            print(f"{intent}: {count} ตัวอย่าง")
            
        print("\nตัวอย่างข้อความสำหรับแต่ละ intent:")
        for intent in intent_counts.keys():
            # หาตัวอย่างข้อความจาก DataFrame
            example = df[df['intent'] == intent]['text'].iloc[0]
            print(f"{intent}: {example}")
        
        print("\nกำลังเตรียมข้อมูล...")
        X, y = preprocessor.prepare_data(texts, intents)
        
        print("กำลังสร้างโมเดล...")
        model = ChatbotModel(
            vocab_size=preprocessor.get_vocab_size(),
            num_classes=preprocessor.get_num_classes(),
            max_len=int(os.getenv('MAX_LENGTH', 50))
        )
        model.build_model()
        model.model.summary()
        
        print("\nกำลังเทรนโมเดล...")
        history = model.train(
            X, y,
            epochs=int(os.getenv('EPOCHS', 100)),
            batch_size=int(os.getenv('BATCH_SIZE', 16))
        )
        
        # สร้างการแสดงผลการเทรน
        visualizer = TrainingVisualizer(history)
        
        # พล็อตกราฟ metrics
        print("\nกำลังสร้างกราฟ...")
        visualizer.plot_training_metrics()
        visualizer.plot_learning_rate()
        
        # ทำนายผลบน training data เพื่อสร้าง confusion matrix
        y_pred = model.predict(X)
        labels = list(preprocessor.label_encoder.classes_)
        visualizer.plot_confusion_matrix(y, y_pred, labels)
        
        # บันทึกสรุปผลการเทรน
        visualizer.save_metrics_summary(y, y_pred, labels)
        
        # บันทึกโมเดล
        model.save_model('models/chatbot_model.h5')
        print("\nบันทึกโมเดลและผลการเทรนเรียบร้อย")
        print("- Model saved to: models/chatbot_model.h5")
        print("- Plots saved to: plots/")
        print("- Training summary: plots/training_summary.txt")
        
    except Exception as e:
        import traceback
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # โหลด environment variables
    load_dotenv('config/.env')
    train()
# utils.py

import json
import os
import re
from datetime import datetime
import logging
from pythainlp.util import normalize

class ChatbotUtils:
    @staticmethod
    def setup_logging(log_dir='logs'):
        """ตั้งค่า logging"""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'chatbot_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    @staticmethod
    def extract_product_info(text):
        """แยกข้อมูลรุ่นและความจุจากข้อความ"""
        # แปลงเป็นตัวพิมพ์เล็กและ normalize
        text = normalize(text.lower())
        
        # ค้นหารุ่น iPhone
        model_pattern = r'iphone\s*(\d{1,2})(?:\s*(pro|pro max|mini|plus))?'
        model_match = re.search(model_pattern, text)
        
        # ค้นหาความจุ
        storage_pattern = r'(\d+)\s*gb'
        storage_match = re.search(storage_pattern, text)
        
        model = None
        if model_match:
            model_num = model_match.group(1)
            model_type = model_match.group(2) or ""
            model = f"iPhone_{model_num}{('_' + model_type.replace(' ', '_')).upper() if model_type else ''}"
            
        storage = f"{storage_match.group(1)}GB" if storage_match else None
        
        return model, storage
    
    @staticmethod
    def load_config(config_path):
        """โหลดไฟล์ config"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_chat_history(chat_id, message, response, intent, filename='chat_history.jsonl'):
        """บันทึกประวัติการสนทนา"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'chat_id': chat_id,
            'message': message,
            'intent': intent,
            'response': response
        }
        
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(history_entry, ensure_ascii=False) + '\n')
    
    @staticmethod
    def analyze_chat_history(filename='chat_history.jsonl'):
        """วิเคราะห์ประวัติการสนทนา"""
        intent_counts = {}
        total_chats = 0
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                intent = entry['intent']
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                total_chats += 1
                
        return {
            'total_chats': total_chats,
            'intent_distribution': {
                intent: count/total_chats 
                for intent, count in intent_counts.items()
            }
        }
    
    @staticmethod
    def format_price(price):
        """จัดรูปแบบราคา"""
        try:
            price = float(str(price).replace(',', ''))
            return f"{price:,.2f} บาท"
        except:
            return str(price)

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """คำนวณ metrics ของโมเดล"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    @staticmethod
    def plot_training_history(history):
        """พล็อตกราฟประวัติการเทรน"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # พล็อต loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # พล็อต accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        return plt
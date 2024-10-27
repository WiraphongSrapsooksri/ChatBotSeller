# model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional # type: ignore
import numpy as np

class ChatbotModel:
    def __init__(self, vocab_size, num_classes, max_len=50, embedding_dim=128):  
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.model = None    

    def build_model(self):
        """สร้างโมเดลแบบ CNN-LSTM ที่ซับซ้อนขึ้น"""
        model = Sequential([
            # Embedding Layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                mask_zero=True  # เพิ่ม mask_zero
            ),
            
            # CNN Layers
            Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
            Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'),
            Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'),
            Dropout(0.2),
            
            # Bidirectional LSTM Layers
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),
            
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            
            # Dense Layers with BatchNormalization
            Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.2),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        # ใช้ optimizer แบบ custom
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=16):
        """ปรับปรุง training process"""
        # Early stopping with more patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4
        )
        
        # Reduce learning rate with more patience
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Add class weights to handle imbalanced data
        class_weights = {
            i: len(y) / (len(np.unique(y)) * sum(y == i))
            for i in np.unique(y)
        }
        
        return self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            class_weight=class_weights,
            shuffle=True
        )    
    

    def predict(self, X):
        """ทำนาย intent"""
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def get_confidence(self, X):
        """คำนวณความเชื่อมั่นในการทำนาย"""
        predictions = self.model.predict(X)
        return np.max(predictions, axis=1)

    def save_model(self, path):
        """บันทึกโมเดล"""
        self.model.save(path, save_format='h5')
    
    def load_model(self, path):
        """โหลดโมเดล"""
        self.model = load_model(path)

# เพิ่มต่อจากโค้ดเดิมใน model.py

class ResponseGenerator:
    def __init__(self, responses_data):
        self.responses = responses_data
        
    def get_response(self, intent, model=None, storage=None):
        """สร้างการตอบกลับตาม intent"""
        if intent not in self.responses:
            return "ขออภัย ไม่เข้าใจคำถาม กรุณาถามใหม่อีกครั้ง"
            
        # กรณีสอบถามราคา
        if intent == "price_inquiry" and model and storage:
            product_data = self.responses["price_inquiry"]
            if model in product_data and storage in product_data[model]:
                price = product_data[model][storage]
                return f"{model.replace('_', ' ')} {storage} ราคา {price}"
            return "ขออภัย ไม่พบข้อมูลราคาสำหรับรุ่นและความจุที่ระบุ"
            
        # กรณีสอบถามสี
        elif intent == "color_inquiry" and model:
            color_data = self.responses["color_inquiry"]
            if model in color_data:
                colors = color_data[model]
                colors_text = ", ".join(colors)
                return f"{model.replace('_', ' ')} มีสีให้เลือกดังนี้: {colors_text}"
            return "ขออภัย ไม่พบข้อมูลสีสำหรับรุ่นที่ระบุ"
            
        # กรณีสอบถามความจุ
        elif intent == "storage_inquiry" and model:
            storage_data = self.responses["storage_inquiry"]
            if model in storage_data:
                storages = storage_data[model]
                storages_text = ", ".join(storages)
                return f"{model.replace('_', ' ')} มีความจุให้เลือกดังนี้: {storages_text}"
            return "ขออภัย ไม่พบข้อมูลความจุสำหรับรุ่นที่ระบุ"
            
        # กรณี greeting
        elif intent == "greeting":
            if isinstance(self.responses[intent], dict) and "responses" in self.responses[intent]:
                responses = self.responses[intent]["responses"]
                return np.random.choice(responses)
            return "สวัสดีครับ/ค่ะ มีอะไรให้ช่วยแนะนำไหมครับ"
            
        # กรณีอื่นๆ
        elif isinstance(self.responses[intent], dict):
            if "responses" in self.responses[intent]:
                responses = self.responses[intent]["responses"]
                return np.random.choice(responses)
            elif "default_response" in self.responses[intent]:
                return self.responses[intent]["default_response"]
                
        return str(self.responses[intent])        
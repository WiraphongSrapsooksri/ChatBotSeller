# scripts/manage_keys.py

import secrets
import os
from pathlib import Path
from datetime import datetime

class APIKeyManager:
    def __init__(self, env_path='config/.env'):
        self.env_path = Path(env_path)
        
    def generate_key(self):
        """สร้าง API key ใหม่"""
        return secrets.token_hex(32)
    
    def update_env_file(self, new_key):
        """อัพเดท API key ในไฟล์ .env"""
        if not self.env_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ {self.env_path}")
            
        # อ่านไฟล์ .env
        with open(self.env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # อัพเดท API_KEY
        with open(self.env_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.startswith('API_KEY='):
                    f.write(f'API_KEY={new_key}\n')
                else:
                    f.write(line)
                    
    def backup_env(self):
        """สำรองไฟล์ .env"""
        if not self.env_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ {self.env_path}")
            
        backup_path = self.env_path.parent / f'.env.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        with open(self.env_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        return backup_path
    
    def rotate_key(self):
        """สร้างและอัพเดท API key ใหม่"""
        # สำรองไฟล์เก่า
        backup_path = self.backup_env()
        
        # สร้าง key ใหม่
        new_key = self.generate_key()
        
        # อัพเดทไฟล์ .env
        self.update_env_file(new_key)
        
        return {
            'new_key': new_key,
            'backup_path': backup_path
        }

def main():
    try:
        manager = APIKeyManager()
        
        # สร้างและอัพเดท key ใหม่
        result = manager.rotate_key()
        
        print("API Key ถูกอัพเดทเรียบร้อย")
        print(f"Key ใหม่: {result['new_key']}")
        print(f"ไฟล์สำรอง: {result['backup_path']}")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    main()



# สคริปต์นี้จะ
# สร้าง API Key ใหม่ที่ปลอดภัย
# สำรองไฟล์ .env เก่า
# อัพเดท API Key ในไฟล์ .env
# แสดง Key ใหม่และที่อยู่ไฟล์สำรอง    
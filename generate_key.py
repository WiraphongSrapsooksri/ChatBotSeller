# generate_key.py
import secrets

def generate_api_key():
    # สร้าง API key ขนาด 32 bytes (256 bits)
    return secrets.token_hex(32)

if __name__ == "__main__":
    api_key = generate_api_key()
    print(f"Your API Key: {api_key}")


#python generate_key.py    
#20ef8b56e557a1e0c255c700aea8ff25eb7a981d70ecb32effa9243899b702fa
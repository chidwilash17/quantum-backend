# backend/app/services/classical_crypto_service.py
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets
import base64

class ClassicalCryptoService:
    def __init__(self):
        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()
        
    def encrypt_classical(self, plaintext):
        """Classical encryption: AES + RSA key wrapping"""
        # Generate AES key
        aes_key = secrets.token_bytes(32)  # 256-bit
        
        # Wrap AES key with RSA
        wrapped_key = self.public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Encrypt data with AES-GCM
        aesgcm = AESGCM(aes_key)
        iv = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(iv, plaintext.encode(), None)
        
        # Split ciphertext and tag
        actual_ciphertext = ciphertext[:-16]
        tag = ciphertext[-16:]
        
        return {
            'aesKey': base64.b64encode(aes_key).decode(),
            'wrappedKey': base64.b64encode(wrapped_key).decode(),
            'encrypted': {
                'iv': base64.b64encode(iv).decode(),
                'ciphertext': base64.b64encode(actual_ciphertext).decode(),
                'tag': base64.b64encode(tag).decode()
            }
        }
        
    def decrypt_classical(self, encrypted_data):
        """Classical decryption"""
        # Unwrap AES key
        wrapped_key = base64.b64decode(encrypted_data['wrappedKey'])
        aes_key = self.private_key.decrypt(
            wrapped_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt with AES-GCM
        aesgcm = AESGCM(aes_key)
        iv = base64.b64decode(encrypted_data['encrypted']['iv'])
        ciphertext = base64.b64decode(encrypted_data['encrypted']['ciphertext'])
        tag = base64.b64decode(encrypted_data['encrypted']['tag'])
        
        full_ciphertext = ciphertext + tag
        
        try:
            plaintext = aesgcm.decrypt(iv, full_ciphertext, None)
            return plaintext.decode()
        except Exception:
            return None

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from cryptography.fernet import Fernet
import base64
import hashlib

router = APIRouter()

class EncryptionRequest(BaseModel):
    plaintext: str
    quantum_key: str

class DecryptionRequest(BaseModel):
    ciphertext: str
    quantum_key: str

def derive_key_from_quantum_key(quantum_key: str) -> bytes:
    """Derive a Fernet-compatible key from quantum key"""
    # Hash the quantum key to get 32 bytes
    digest = hashlib.sha256(quantum_key.encode()).digest()
    # Encode as base64 for Fernet
    return base64.urlsafe_b64encode(digest)

@router.post("/encrypt")
async def encrypt_message(request: EncryptionRequest):
    """Encrypt message using quantum-derived key"""
    try:
        # Derive encryption key from quantum key
        derived_key = derive_key_from_quantum_key(request.quantum_key)
        fernet = Fernet(derived_key)
        
        # Encrypt the message
        encrypted = fernet.encrypt(request.plaintext.encode())
        ciphertext = base64.b64encode(encrypted).decode()
        
        return {
            "success": True,
            "ciphertext": ciphertext,
            "key_length": len(request.quantum_key),
            "algorithm": "AES-128 with quantum-derived key"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

@router.post("/decrypt")
async def decrypt_message(request: DecryptionRequest):
    """Decrypt message using quantum-derived key"""
    try:
        # Derive encryption key from quantum key
        derived_key = derive_key_from_quantum_key(request.quantum_key)
        fernet = Fernet(derived_key)
        
        # Decrypt the message
        encrypted = base64.b64decode(request.ciphertext.encode())
        decrypted = fernet.decrypt(encrypted)
        plaintext = decrypted.decode()
        
        return {
            "success": True,
            "plaintext": plaintext,
            "key_length": len(request.quantum_key),
            "algorithm": "AES-128 with quantum-derived key"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")

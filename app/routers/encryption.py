from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import random
import base64
import os

router = APIRouter()

class EncryptRequest(BaseModel):
    message: str
    eavesdropperActive: bool = False

@router.post("/qkd-encrypt")
async def qkd_encrypt(request: EncryptRequest):
    """QKD + AES-GCM Architecture"""
    print(f"🔍 QKD endpoint called: {request.message}")
    try:
        # Simulate BB84 QKD process
        raw_key = ''.join([str(random.randint(0, 1)) for _ in range(16)])
        reconciled_key = ''.join([str(random.randint(0, 1)) for _ in range(16)])
        
        # Simulate QBER based on eavesdropper
        if request.eavesdropperActive:
            qber = random.uniform(15, 35)  # High QBER with eavesdropper
        else:
            qber = random.uniform(1, 8)    # Low QBER without eavesdropper
            
        # If QBER too high, refuse connection
        if qber > 11:
            raise HTTPException(
                status_code=400, 
                detail=f"QKD failed - QBER too high: {qber:.1f}%. Eavesdropper detected!"
            )
        
        # Generate final key
        final_key = base64.b64encode(os.urandom(32)).decode()
        
        # Simulate AES-GCM encryption
        iv = base64.b64encode(os.urandom(12)).decode()
        ciphertext = base64.b64encode((request.message.encode() + os.urandom(16))).decode()
        tag = base64.b64encode(os.urandom(16)).decode()
        
        return {
            'qkd': {
                'rawKey': raw_key,
                'reconciledKey': reconciled_key,
                'qber': round(qber, 2)
            },
            'finalKey': final_key,
            'encrypted': {
                'iv': iv,
                'ciphertext': ciphertext,
                'tag': tag
            },
            'decrypted': request.message,  # Simulate successful decryption
            'originalMessage': request.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classical-encrypt")
async def classical_encrypt(request: EncryptRequest):
    """RSA + AES-GCM Architecture"""
    print(f"🔍 Classical endpoint called: {request.message}")
    try:
        # Generate AES key
        aes_key = base64.b64encode(os.urandom(32)).decode()
        
        # Simulate RSA key wrapping
        wrapped_key = base64.b64encode(os.urandom(256)).decode()
        
        # Simulate AES-GCM encryption
        iv = base64.b64encode(os.urandom(12)).decode()
        ciphertext = base64.b64encode((request.message.encode() + os.urandom(16))).decode()
        tag = base64.b64encode(os.urandom(16)).decode()
        
        return {
            'aesKey': aes_key,
            'wrappedKey': wrapped_key,
            'encrypted': {
                'iv': iv,
                'ciphertext': ciphertext,
                'tag': tag
            },
            'decrypted': request.message,  # Simulate successful decryption
            'originalMessage': request.message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

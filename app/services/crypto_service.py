# backend/app/services/crypto_service.py
import secrets
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import base64

class QKDCryptoService:
    def __init__(self):
        self.aesgcm = None
        
    def simulate_bb84_qkd(self, key_length=256, eavesdropper_present=False):
        """Simulate BB84 protocol with optional eavesdropping"""
        # Generate random bits and bases
        alice_bits = [secrets.randbits(1) for _ in range(key_length * 2)]
        alice_bases = [secrets.randbits(1) for _ in range(key_length * 2)]
        bob_bases = [secrets.randbits(1) for _ in range(key_length * 2)]
        
        # Simulate measurement (Bob's results)
        bob_bits = []
        qber = 0
        
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:  # Same basis
                if eavesdropper_present and secrets.randbits(2) == 0:  # 25% error from Eve
                    bob_bits.append(1 - alice_bits[i])  # Flip bit
                    qber += 1
                else:
                    bob_bits.append(alice_bits[i])  # Correct bit
            else:  # Different basis - 50% chance of error
                bob_bits.append(secrets.randbits(1))
                
        # Sifting - keep only matching bases
        sifted_key = []
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])
        
        # Calculate QBER
        if len(sifted_key) > 0:
            qber_percentage = (qber / len(sifted_key)) * 100
        else:
            qber_percentage = 0
            
        # Privacy amplification using HKDF
        raw_key_bytes = bytes(sifted_key[:32])  # Take first 32 bits as bytes
        
        if qber_percentage > 11:  # Security threshold
            return None, qber_percentage  # Refuse to establish key
            
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=None,
            info=b'QKD-derived-key',
        )
        
        final_key = hkdf.derive(raw_key_bytes)
        
        return {
            'rawKey': ''.join(map(str, alice_bits[:16])),  # Show first 16 bits
            'reconciledKey': ''.join(map(str, sifted_key[:16])),
            'finalKey': base64.b64encode(final_key).decode(),
            'qber': round(qber_percentage, 2)
        }, qber_percentage
        
    def encrypt_with_aes_gcm(self, plaintext, key_b64):
        """Encrypt using AES-GCM"""
        key = base64.b64decode(key_b64)
        aesgcm = AESGCM(key)
        
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        ciphertext = aesgcm.encrypt(iv, plaintext.encode(), None)
        
        # Split ciphertext and tag (last 16 bytes)
        actual_ciphertext = ciphertext[:-16]
        tag = ciphertext[-16:]
        
        return {
            'iv': base64.b64encode(iv).decode(),
            'ciphertext': base64.b64encode(actual_ciphertext).decode(),
            'tag': base64.b64encode(tag).decode()
        }
        
    def decrypt_with_aes_gcm(self, encrypted_data, key_b64):
        """Decrypt using AES-GCM"""
        key = base64.b64decode(key_b64)
        aesgcm = AESGCM(key)
        
        iv = base64.b64decode(encrypted_data['iv'])
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        # Combine ciphertext and tag
        full_ciphertext = ciphertext + tag
        
        try:
            plaintext = aesgcm.decrypt(iv, full_ciphertext, None)
            return plaintext.decode()
        except Exception:
            return None  # Authentication failed

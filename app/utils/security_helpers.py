import hashlib
import hmac
import secrets
from typing import List, Dict, Tuple
import numpy as np

def calculate_mutual_information(alice_key: List[int], bob_key: List[int], eve_key: List[int]) -> Dict[str, float]:
    """Calculate mutual information for security analysis"""
    
    def entropy(data: List[int]) -> float:
        if not data:
            return 0.0
        _, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        return -np.sum(probs * np.log2(probs + 1e-10))  # Add small epsilon for numerical stability
    
    def joint_entropy(data1: List[int], data2: List[int]) -> float:
        if len(data1) != len(data2) or not data1:
            return 0.0
        pairs = [(d1, d2) for d1, d2 in zip(data1, data2)]
        _, counts = np.unique(pairs, return_counts=True, axis=0)
        probs = counts / len(pairs)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    h_alice = entropy(alice_key)
    h_bob = entropy(bob_key)
    h_eve = entropy(eve_key)
    
    h_alice_bob = joint_entropy(alice_key, bob_key)
    h_alice_eve = joint_entropy(alice_key, eve_key)
    
    # Mutual information I(A;B) = H(A) + H(B) - H(A,B)
    i_alice_bob = h_alice + h_bob - h_alice_bob
    i_alice_eve = h_alice + h_eve - h_alice_eve
    
    return {
        "mutual_info_alice_bob": float(i_alice_bob),
        "mutual_info_alice_eve": float(i_alice_eve),
        "security_ratio": float(i_alice_bob / (i_alice_eve + 1e-10))
    }

def privacy_amplification(raw_key: List[int], target_length: int) -> List[int]:
    """Apply privacy amplification using universal hashing"""
    if len(raw_key) <= target_length:
        return raw_key
    
    # Convert to bytes
    raw_bytes = bytes([sum(raw_key[i:i+8][j] << (7-j) 
                          for j in range(min(8, len(raw_key[i:i+8])))) 
                      for i in range(0, len(raw_key), 8)])
    
    # Use SHA-256 for privacy amplification
    hash_obj = hashlib.sha256(raw_bytes)
    amplified_bytes = hash_obj.digest()
    
    # Convert back to bits
    amplified_bits = []
    for byte in amplified_bytes:
        for i in range(8):
            amplified_bits.append((byte >> (7-i)) & 1)
    
    return amplified_bits[:target_length]

def error_correction_syndrome(key: List[int], block_size: int = 7) -> List[int]:
    """Calculate syndrome for simple error correction"""
    syndromes = []
    
    for i in range(0, len(key), block_size):
        block = key[i:i+block_size]
        if len(block) < block_size:
            block.extend([0] * (block_size - len(block)))  # Pad with zeros
        
        # Simple parity check syndrome
        syndrome = sum(block) % 2
        syndromes.append(syndrome)
    
    return syndromes

def detect_eavesdropping(alice_key: List[int], bob_key: List[int], 
                        threshold: float = 0.11) -> Dict[str, any]:
    """Detect potential eavesdropping based on error rate"""
    if not alice_key or not bob_key or len(alice_key) != len(bob_key):
        return {
            "eavesdropping_detected": True,
            "error_rate": 1.0,
            "confidence": 1.0,
            "recommendation": "ABORT - Key length mismatch"
        }
    
    error_rate = sum(a != b for a, b in zip(alice_key, bob_key)) / len(alice_key)
    
    # Statistical test for eavesdropping
    n = len(alice_key)
    expected_errors = n * 0.005  # Expected natural error rate
    actual_errors = n * error_rate
    
    # Simple z-test approximation
    if expected_errors > 0:
        z_score = abs(actual_errors - expected_errors) / np.sqrt(expected_errors)
        confidence = min(1.0, z_score / 2.0)  # Rough confidence measure
    else:
        confidence = 1.0 if error_rate > 0 else 0.0
    
    return {
        "eavesdropping_detected": error_rate > threshold,
        "error_rate": float(error_rate),
        "confidence": float(confidence),
        "recommendation": "ABORT - Possible eavesdropping" if error_rate > threshold else "PROCEED"
    }

def generate_authentication_tag(key: List[int], message: str) -> str:
    """Generate HMAC authentication tag"""
    key_bytes = bytes([sum(key[i:i+8][j] << (7-j) 
                          for j in range(min(8, len(key[i:i+8])))) 
                      for i in range(0, len(key), 8)])
    
    return hmac.new(key_bytes, message.encode(), hashlib.sha256).hexdigest()

def verify_authentication_tag(key: List[int], message: str, tag: str) -> bool:
    """Verify HMAC authentication tag"""
    expected_tag = generate_authentication_tag(key, message)
    return hmac.compare_digest(expected_tag, tag)

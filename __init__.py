# At the top of tests/backend/test_services.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import asyncio
from backend.app.services.bb84_service import BB84Service
from backend.app.services.quantum_service import QuantumService
from backend.app.services.crypto_service import CryptoService

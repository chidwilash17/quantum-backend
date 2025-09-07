import numpy as np
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def binary_to_decimal(binary_list: List[int]) -> int:
    """Convert binary list to decimal"""
    return int(''.join(map(str, binary_list)), 2)

def decimal_to_binary(decimal: int, length: int) -> List[int]:
    """Convert decimal to binary list of specified length"""
    binary_str = format(decimal, f'0{length}b')
    return [int(bit) for bit in binary_str]

def calculate_fidelity(state1: Statevector, state2: Statevector) -> float:
    """Calculate quantum state fidelity"""
    return float(np.abs(state1.conjugate().dot(state2.data))**2)

def pauli_basis_measurement(qubit_state: Statevector, basis: str) -> Tuple[int, float]:
    """
    Simulate measurement in Pauli basis
    basis: 'X', 'Y', or 'Z'
    Returns: (measurement_result, probability)
    """
    if basis == 'Z':
        # Computational basis measurement
        probs = qubit_state.probabilities()
        result = np.random.choice([0, 1], p=probs)
        return result, probs[result]
    
    elif basis == 'X':
        # X-basis measurement (diagonal basis)
        # Apply Hadamard before Z measurement
        qc = QuantumCircuit(1)
        qc.initialize(qubit_state, 0)
        qc.h(0)
        new_state = Statevector.from_instruction(qc)
        probs = new_state.probabilities()
        result = np.random.choice([0, 1], p=probs)
        return result, probs[result]
    
    elif basis == 'Y':
        # Y-basis measurement
        qc = QuantumCircuit(1)
        qc.initialize(qubit_state, 0)
        qc.sdg(0)  # S†
        qc.h(0)    # H
        new_state = Statevector.from_instruction(qc)
        probs = new_state.probabilities()
        result = np.random.choice([0, 1], p=probs)
        return result, probs[result]
    
    else:
        raise ValueError(f"Unknown basis: {basis}")

def calculate_bloch_coordinates(theta: float, phi: float) -> Tuple[float, float, float]:
    """Calculate Bloch sphere coordinates from angles"""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return float(x), float(y), float(z)

def estimate_error_rate(alice_key: List[int], bob_key: List[int]) -> float:
    """Estimate quantum bit error rate between two keys"""
    if len(alice_key) != len(bob_key) or len(alice_key) == 0:
        return 1.0
    
    errors = sum(a != b for a, b in zip(alice_key, bob_key))
    return errors / len(alice_key)

def apply_quantum_channel_noise(circuit: QuantumCircuit, error_rate: float) -> QuantumCircuit:
    """Apply noise to quantum circuit"""
    noisy_circuit = circuit.copy()
    
    # Add depolarizing noise with given error rate
    for i in range(circuit.num_qubits):
        if np.random.random() < error_rate:
            # Apply random Pauli error
            error_type = np.random.choice(['I', 'X', 'Y', 'Z'])
            if error_type == 'X':
                noisy_circuit.x(i)
            elif error_type == 'Y':
                noisy_circuit.y(i)
            elif error_type == 'Z':
                noisy_circuit.z(i)
            # 'I' means no error
    
    return noisy_circuit

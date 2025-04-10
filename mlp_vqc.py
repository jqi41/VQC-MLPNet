#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 18:14:21 2025

@author: junqi
"""

import torch
from torch import nn

# Torch Quantum
import torchquantum as tq
import torchquantum.functional as tqf

seed = 1234
torch.manual_seed(seed)

####### Detect if running on the clusters #######
# Using CUDA, MPS, or GPU: 
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use CUDA if available
    print("is CUDA available?", torch.cuda.is_available())
else:
    device = torch.device("cpu")    # Fallback to CPU
    print("Running on the CPU")


class MLP_VQC(tq.QuantumModule):
    """
    MLP_VQC model with a realistic IBM noise simulation.
    
    The model encodes classical weights into quantum states via amplitude encoding,
    applies variational layers (rotations and CNOT entanglement), and then injects two types 
    of noise to mimic IBM hardware:
    
      - Amplitude damping noise (T₁ decay): With probability amplitude_damping_rate, the qubit is reset.
      - Phase damping noise (T₂ dephasing): With probability phase_damping_rate, a Z gate is applied.
    
    These noise channels are applied after the entanglement block in each variational layer.
    """
    def __init__(self,
                 n_wires: int = 8,
                 n_qlayers: int = 1,
                 input_dims: int = 16,
                 hidden_units: int = 128,
                 out_features: int = 2,
                 add_fc: bool = True,
                 amplitude_damping_rate: float = 0.01,
                 phase_damping_rate: float = 0.01):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.hidden_units = hidden_units
        self.amplitude_damping_rate = amplitude_damping_rate
        self.phase_damping_rate = phase_damping_rate
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        
        # Initialize weights (non-trainable in this context)
        self.W1 = torch.randn(input_dims, hidden_units, requires_grad=False)
        self.W2 = nn.Linear(hidden_units, out_features)
        self.encoder = tq.AmplitudeEncoder()
        
        # Define trainable quantum parameters (rotational gates)
        self.params = nn.ModuleDict({
            f"layer_{k}_wire_{i}": nn.ModuleDict({
                "rx": tq.RX(has_params=True, trainable=True),
                "ry": tq.RY(has_params=True, trainable=True),
                "rz": tq.RZ(has_params=True, trainable=True)
            })
            for k in range(n_qlayers) for i in range(n_wires)
        })
        
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.add_fc = add_fc
        if add_fc:
            self.fc_layer = nn.Linear(self.n_wires, self.hidden_units)
    
    def apply_ibm_noise(self):
        """
        Apply IBM-like noise:
          - Amplitude damping: with probability amplitude_damping_rate, reset the qubit (simulate T₁ decay).
          - Phase damping: with probability phase_damping_rate, apply a Z gate (simulate T₂ dephasing).
        """
        for i in range(self.n_wires):
            # Simulate amplitude damping noise:
            if torch.rand(1).item() < self.amplitude_damping_rate:
                # Here we assume tqf.reset exists and resets the qubit state.
                tqf.reset(self.q_device, wires=i)
            # Simulate phase damping noise:
            if torch.rand(1).item() < self.phase_damping_rate:
                tqf.z(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
    
    @tq.static_support
    def forward(self, 
                x: torch.Tensor,
                q_device: tq.QuantumDevice,
                is_train: bool = True,
                W1=None,
                device=torch.device("cpu")):
        self.q_device = q_device
        if is_train:
            W1 = W1 if W1 is not None else self.W1
            # Encode classical weights into the quantum state via amplitude encoding.
            self.encoder(self.q_device, self.W1)
            
            # Apply variational layers:
            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    gates = self.params[f"layer_{k}_wire_{i}"]
                    gates["rx"](self.q_device, wires=i)
                    gates["ry"](self.q_device, wires=i)
                    gates["rz"](self.q_device, wires=i)
                # Apply a CNOT chain for entanglement.
                for i in range(self.n_wires - 1):
                    tqf.cnot(self.q_device, wires=[i, i + 1], static=self.static_mode, parent_graph=self.graph)
                tqf.cnot(self.q_device, wires=[self.n_wires - 1, 0], static=self.static_mode, parent_graph=self.graph)
                
                # Inject IBM-like noise after entanglement.
                self.apply_ibm_noise()
            
            qc_out = self.measure(self.q_device)
            if self.add_fc:
                qc_out = self.fc_layer(qc_out)
            h = torch.relu(torch.matmul(x, qc_out))
            out = self.W2(h)
            return out, qc_out
        else:
            h = torch.relu(torch.matmul(x, W1))
            out = self.W2(h)
            return out

    def reset_quantum_device(self, bsz):
        self.q_device.reset_states(bsz)
    
    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

# Example usage
if __name__ == "__main__":
    
    test_signal = torch.randn(500, 16, device=device)
    test_signal_label = torch.randint(low=0, high=2, size=(500,), device=device)
    dev = tq.QuantumDevice(n_wires=8, bsz=test_signal.shape[0])
    mlp_mps_vqc = MLP_VQC(n_wires=8, n_qlayers=2, hidden_units=256, input_dims=16, 
                          amplitude_damping_rate = 0.001,
                          phase_damping_rate = 0.001, 
                          add_fc=True, 
                          out_features=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_mps_vqc.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # For each 100 iterations, the lr is divided by 2

    W1 = torch.randn(16, 1024, requires_grad=False, device=device)
    
    for epoch in range(500):
        optimizer.zero_grad()
        
        out, W1 = mlp_mps_vqc(test_signal, q_device=dev, is_train=True, W1=W1)
        loss = criterion(out, test_signal_label)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f'Epoch {epoch+1} loss = {loss.item():.4f}')
        
    with torch.no_grad():
        out = mlp_mps_vqc(test_signal, q_device=dev, is_train=False, W1=W1.detach())
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == test_signal_label).sum().item()
        accuracy = correct / test_signal_label.size(0)
        
        print(f'Test Accuracy: {accuracy:.4f}')

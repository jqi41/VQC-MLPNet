#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 09:20:50 2023

@author: junqi
"""

import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf


class VQC(tq.QuantumModule):
    def __init__(self, n_wires=8, n_qlayers=1, tensor_product_enc=True, add_fc=False, out_features=2, noise_prob=0.01):
        """
        noise_prob: probability for depolarizing noise on each qubit.
        """
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.add_fc = add_fc
        self.noise_prob = noise_prob

        # Encoder: either tensor product (using RY per wire) or amplitude encoding.
        self.encoder = (
            tq.GeneralEncoder([{'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
            if tensor_product_enc else tq.AmplitudeEncoder()
        )

        # Variational parameters stored in a ModuleDict.
        self.params = nn.ModuleDict({
            f"layer_{k}_wire_{i}": nn.ModuleDict({
                "rx": tq.RX(has_params=True, trainable=True),
                "ry": tq.RY(has_params=True, trainable=True),
                "rz": tq.RZ(has_params=True, trainable=True)
            })
            for k in range(n_qlayers) for i in range(n_wires)
        })

        # Measurement layer: measure all qubits in the Pauli-Z basis.
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Optional fully connected layer for classical post-processing.
        if add_fc:
            self.fc_layer = nn.Linear(n_wires, out_features)

    def reset_quantum_device(self, bsz: int):
        """Reset the quantum device states for a given batch size."""
        self.q_device.reset_states(bsz)

    def apply_variational_layer(self, layer_idx: int):
        """Apply parameterized single-qubit rotations for a given layer index."""
        for i in range(self.n_wires):
            gates = self.params[f"layer_{layer_idx}_wire_{i}"]
            gates["rx"](self.q_device, wires=i)
            gates["ry"](self.q_device, wires=i)
            gates["rz"](self.q_device, wires=i)

    def apply_entanglement(self):
        """Default entanglement: a ring of CNOTs with periodic boundary conditions."""
        for i in range(self.n_wires - 1):
            tqf.cnot(self.q_device, wires=[i, i + 1],
                     static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(self.q_device, wires=[self.n_wires - 1, 0],
                 static=self.static_mode, parent_graph=self.graph)

    def apply_depolarizing_noise(self):
        """
        Custom depolarizing noise: For each qubit, with probability noise_prob, apply a random Pauli error.
        """
        for i in range(self.n_wires):
            # For each qubit, decide whether to apply noise.
            if torch.rand(1).item() < self.noise_prob:
                error_type = torch.randint(0, 3, (1,)).item()
                if error_type == 0:
                    tqf.x(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif error_type == 1:
                    tqf.y(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif error_type == 2:
                    tqf.z(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)

    @tq.static_support
    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.q_device = q_device
        # Reset the quantum states for the current batch.
        self.reset_quantum_device(x.shape[0])
        # Encode classical data into quantum states.
        self.encoder(self.q_device, x)
        # For each variational layer: apply rotations, entanglement, then noise.
        for k in range(self.n_qlayers):
            self.apply_variational_layer(k)
            self.apply_entanglement()
            self.apply_depolarizing_noise()  # Inject noise at each layer.
        # Measure the quantum state.
        qc_out = self.measure(self.q_device)
        return self.fc_layer(qc_out) if self.add_fc else qc_out

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))


class MPS_VQC(VQC):
    """
    A VQC variant with a Matrix Product State (MPS) structure.
    Here the entanglement (via CNOT gates) is applied immediately after the rotations
    for each qubit (except the last one) within the variational layer.
    This modified version also injects depolarizing noise after each layer.
    """
    def __init__(self, n_wires=8, n_qlayers=1, tensor_product_enc=True, add_fc=False, out_features=2, noise_prob=0.01):
        # Call parent constructor
        super().__init__(n_wires, n_qlayers, tensor_product_enc, add_fc, out_features)
        self.noise_prob = noise_prob

    def apply_depolarizing_noise(self):
        """Custom depolarizing noise injection: for each qubit, with probability noise_prob, apply a random Pauli error."""
        for i in range(self.n_wires):
            if torch.rand(1).item() < self.noise_prob:
                error_type = torch.randint(0, 3, (1,)).item()
                if error_type == 0:
                    tqf.x(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif error_type == 1:
                    tqf.y(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif error_type == 2:
                    tqf.z(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)

    @tq.static_support
    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.q_device = q_device
        self.reset_quantum_device(x.shape[0])
        self.encoder(self.q_device, x)
        # For each variational layer, apply rotations followed by nearest-neighbor entanglement.
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                gates = self.params[f"layer_{k}_wire_{i}"]
                gates["rx"](self.q_device, wires=i)
                gates["ry"](self.q_device, wires=i)
                gates["rz"](self.q_device, wires=i)
                if i < self.n_wires - 1:
                    tqf.cnot(self.q_device, wires=[i, i + 1], 
                             static=self.static_mode, parent_graph=self.graph)
            # Inject depolarizing noise after each layer.
            self.apply_depolarizing_noise()
        qc_out = self.measure(self.q_device)
        return self.fc_layer(qc_out) if self.add_fc else qc_out

    def reset_quantum_device(self, bsz: int):
        self.q_device.reset_states(bsz)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))


class TTN_VQC(VQC):
    """
    A VQC variant with a Tensor Tree Network (TTN) structure.
    The entanglement pattern follows a tree (binary tree) structure.
    This modified version also injects depolarizing noise after the rotations
    and after the TTN entanglement pattern.
    """
    def __init__(self, n_wires=8, n_qlayers=1, tensor_product_enc=True, add_fc=False, out_features=2, noise_prob=0.01):
        super().__init__(n_wires, n_qlayers, tensor_product_enc, add_fc, out_features)
        self.noise_prob = noise_prob

    def apply_depolarizing_noise(self):
        """Custom depolarizing noise injection: for each qubit, with probability noise_prob, apply a random Pauli error."""
        for i in range(self.n_wires):
            if torch.rand(1).item() < self.noise_prob:
                error_type = torch.randint(0, 3, (1,)).item()
                if error_type == 0:
                    tqf.x(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif error_type == 1:
                    tqf.y(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif error_type == 2:
                    tqf.z(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)

    def apply_ttn_entanglement(self):
        """Apply entanglement gates according to a TTN structure."""
        half = self.n_wires // 2
        for i in range(half):
            parent = i
            child = i + half
            tqf.cnot(self.q_device, wires=[parent, child], 
                     static=self.static_mode, parent_graph=self.graph)
            # Additional entanglement layers for deeper connections
            for layer in range(1, self.n_qlayers):
                tqf.cnot(self.q_device, wires=[parent, (parent + 2**layer) % self.n_wires],
                         static=self.static_mode, parent_graph=self.graph)
                tqf.cnot(self.q_device, wires=[child, (child + 2**layer) % self.n_wires],
                         static=self.static_mode, parent_graph=self.graph)

    @tq.static_support
    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.q_device = q_device
        self.reset_quantum_device(x.shape[0])
        self.encoder(self.q_device, x)
        # Apply variational layers (rotations) with noise injected after each layer.
        for k in range(self.n_qlayers):
            self.apply_variational_layer(k)
            self.apply_depolarizing_noise()
        # Apply the TTN entanglement pattern.
        self.apply_ttn_entanglement()
        # Inject noise after TTN entanglement.
        self.apply_depolarizing_noise()
        qc_out = self.measure(self.q_device)
        return self.fc_layer(qc_out) if self.add_fc else qc_out

    def reset_quantum_device(self, bsz: int):
        self.q_device.reset_states(bsz)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Create sample input signals and labels
    test_signal = torch.randn(500, 16, device=device)
    test_signal_label = torch.randint(0, 2, (500,), device=device)

    # Initialize a quantum device with 8 wires and a batch size equal to the number of samples
    dev = tq.QuantumDevice(n_wires=8, bsz=test_signal.shape[0])
    
    # Instantiate one of the models – here we use the MPS variant with a final fully connected layer
    vqc = VQC(n_wires=8, n_qlayers=2, tensor_product_enc=False, add_fc=True,
              out_features=2, noise_prob=0.01).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vqc.parameters(), lr=0.01)

    # Training loop
    for epoch in range(200):
        optimizer.zero_grad()
        out = vqc(test_signal, q_device=dev)
        loss = criterion(out, test_signal_label)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}: Loss = {loss.item():.4f}')

    # Evaluate on the training data
    with torch.no_grad():
        out = vqc(test_signal, q_device=dev)
        predicted = torch.argmax(out, dim=1)
        accuracy = (predicted == test_signal_label).float().mean().item()
        print(f'Test Accuracy: {accuracy:.4f}')
    
## VQC-MLPNet: Integrating VQCs with MLPs for Scalable and Noise-robust Quantum Machine Learning
The implementation of A Novel Hybrid quantum-classical Architecture with An Interplay Between MLP and VQC

Our codes include the experiments of VQC-MLPNet (mlp_vqc.py) and its experimental running file (mlp_vqc_exp.py)

The simulations of TTN-VQC can be referred to our previous repo: https://github.com/jqi41/TTN-VQC and https://github.com/jqi41/Pretrained-TTN-VQC

### Installation 

The main dependencies include *pytorch* and *torchquantum*

#### Torch Quantum 
```
pip3 install torchquantum
```

 ### 0. Downloading the dataset
```
git clone https://gitlab.com/QMAI/mlqe_2023_edx.git
```

### 1. Simulating VQC_MLPNet experiments

#### 1.1 Assessing the representation power (Noise-free)
python mlp_vqc.py --num_qubits=12 --depth_vqc=6 --lr=0.01 --batch_size=8 --amplitude_dampling_rate=0.0 --phase_dampling_rate=0.0 --test_kind='rep'

#### 1.2 Assessing the generalization power (Noise-free)
python mlp_vqc.py --num_qubits=12 --depth_vqc=6 --lr=0.01 --batch_size=8 --amplitude_dampling_rate=0.0 --phase_dampling_rate=0.0 --test_kind='gen'

### 1.3 Experimental simulation with IBM Quantum noises (ADR=0.05, PDR=0.05)
python mlp_vqc.py --num_qubits=12 --batch_size=8 --lr=0.01 --amplitude_dampling_rate=0.05 --phase_dampling_rate=0.05 --test_kind='gen'

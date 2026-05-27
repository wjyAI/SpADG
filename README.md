
Code and appendix for the KDD 2026 paper: Sparse Additive Models for Domain Generalization.

##  Directory Structure

```
SpADG/
├── README.md                 # Project documentation
├── Appendix.pdf              # Appendix for the paper
├── synthetic/                # Synthetic dataset experiments
│   ├── SpADG.py              # SpADG algorithm implementation
│   ├── main_kernel.py        # Kernel-based experiment
│   └── main_nn.py            # Neural network-based experiment
├── HAR/                      # Human Activity Recognition experiments
│   ├── SpADG.py              # SpADG algorithm implementation
│   ├── main_kernel.py        # Kernel-based experiment
│   └── main_nn.py            # Neural network-based experiment
└── Park/                     # Parkinson's disease regression experiments
    ├── SpADG.py              # SpADG algorithm implementation
    ├── main_kernel.py        # Kernel-based experiment (regression)
    └── main_nn.py            # Neural network-based experiment (regression)
```

##  Running Experiments

### Synthetic Dataset
```bash
cd synthetic
python main_kernel.py   # Run kernel-based SpADG
python main_nn.py       # Run neural network-based SpADG
```

### HAR Dataset
```bash
cd HAR
python main_kernel.py   # Run kernel-based SpADG
python main_nn.py       # Run neural network-based SpADG
```

### Parkinson's Dataset
```bash
cd Park
python main_kernel.py   # Run kernel-based SpADG (regression)
python main_nn.py       # Run neural network-based SpADG (regression)
```

##  Experimental Setup (default)

### Synthetic Dataset
- **Data Generation**: Elliptical distributions with random rotations
- **Domains**: 40 total (30 train, 10 test)
- **Samples per domain**: 32
- **Features**: 50 (User-definable)

### [HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Source**: UCI Human Activity Recognition Dataset
- **Binary task**: Walking (class 2) vs Walking Upstairs (class 3)
- **Domains**: 30 total (20 train, 5 val, 5 test)
- **Samples per domain**: 50
- **Features**: 561

### [Parkinson's Telemonitoring Dataset](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
- **Source**: UCI Parkinson's Telemonitoring Dataset
- **Task**: Regression (motor UPDRS or total UPDRS score prediction)
- **Domains**: 42 total (25 train, 5 val, 10 test)
- **Samples per domain**: 10
- **Features**: 19

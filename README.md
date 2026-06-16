# Quantum Interference Transformer (QIT)

A fully quantum sequence architecture where attention emerges from amplitude interference and entanglement rather than classical dot-product similarity.

> **Central claim:** sequence intelligence can emerge from interference dynamics instead of classical attention matrices.

**Status:** QIT-0 prototype — parity task benchmark complete.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Research Paper Draft](paper/QIT_paper_draft.md) | Full academic paper: architecture, experiments, results, future work |
| [Software Engineer's Guide](paper/QIT_engineer_guide.md) | Quantum computing explained through programming analogies — from qubits to the full circuit |
| [Blueprint](QIT_Research_Blueprint.md) | Original research concept and 30-day roadmap |

---

## Results at a Glance

Benchmark: 4-bit parity task, all 16 inputs, Adam lr=0.05.

| Model | Params | Epochs to 99% | ms / epoch |
|-------|--------|---------------|------------|
| **QIT-0** | **78** | **3** | 188 ms (simulator) |
| MLP | 94 | 36 | 0.7 ms |
| GRU | 110 | 53 | 1.1 ms |
| Transformer | 206 | 175 | 2.2 ms |

QIT-0 converges **12–58× faster** with fewer parameters than any baseline. The per-epoch overhead is the cost of classical simulation of 2⁸ = 256 quantum amplitudes, not a property of the algorithm.

---

## Quick Start

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and set up the environment

```bash
git clone <repo-url>
cd qit
uv sync          # creates .venv and installs all dependencies
```

Dependencies installed: `pennylane`, `torch`, `numpy`, `matplotlib`, `scikit-learn`.

### 3. Train QIT-0 on the parity task

```bash
uv run python experiments/train_qit0.py
```

Expected output:
```
QIT-0  |  Parity Task
  qubits : 4 tokens × 2 qubits = 8 total
  layers : 2
  ...
Epoch  Train Loss  Train Acc  Test Loss  Test Acc    LR   Time
    1      0.68       0.50       0.95      0.50    0.05   0.2s
   ...
   17      0.23       1.00       0.57      1.00    0.025  0.1s

Target accuracy 95% reached — stopping at epoch 17.
```

Results (loss/accuracy curves + epoch timing) are saved to `results/qit0_parity.png`.

### 4. Run the full benchmark

```bash
uv run python benchmarks/compare.py
```

Trains QIT-0, MLP, GRU, and Transformer on the same dataset and produces a comparison table + `results/benchmark.png`.

---

## Project Structure

```
qit/
│
├── qit/                        Core package
│   ├── model.py                QIT — full pipeline (Embedding → QIA → Decoder)
│   ├── measurement.py          ClassicalDecoder — quantum→classical boundary
│   ├── attention/
│   │   └── interference.py     QuantumInterferenceAttention (nn.Module)
│   ├── encoding/               Token → qubit state
│   │   ├── angle.py            RY(x) per qubit          ← default
│   │   ├── phase.py            H + RZ(x) per qubit
│   │   ├── amplitude.py        AmplitudeEmbedding
│   │   └── basis.py            BasisEmbedding (integers)
│   └── layers/                 Quantum circuit subroutines
│       ├── entangle.py         u_entangle — ring CNOT across token registers
│       ├── mix.py              u_mix — BasicEntanglerLayers
│       └── memory.py           u_memory — per-qubit Rot (residual, future use)
│
├── tasks/
│   └── parity.py               ParityDataset + loader factories
│
├── baselines/                  Classical comparison models
│   ├── mlp.py                  MLP (94 params)
│   ├── rnn.py                  GRU (110 params)
│   └── transformer.py          Transformer (206 params)
│
├── experiments/
│   └── train_qit0.py           QIT-0 training script
│
├── benchmarks/
│   └── compare.py              QIT-0 vs all baselines
│
├── results/                    Generated outputs (gitignored)
│   ├── qit0_parity.png
│   └── benchmark.png
│
├── paper/
│   ├── QIT_paper_draft.md      Research paper draft
│   └── QIT_engineer_guide.md   SE-perspective learning guide
│
├── diagrams/                   Architecture diagrams (planned)
├── notebooks/                  Analysis notebooks (planned)
└── pyproject.toml
```

---

## Architecture

```
Classical Tokens
      ↓
nn.Embedding  →  tanh(·) × π  →  angle features
      ↓
QuantumInterferenceAttention
  ├── Angle Encoding      RY(xᵢ) per qubit
  ├── U_entangle          ring CNOT across token registers
  ├── U_attention         StronglyEntanglingLayers (learned interference)
  └── U_mix               BasicEntanglerLayers
      ↓  ⟨Zᵢ⟩ × 8 qubits
ClassicalDecoder  →  logits
```

**U_QIT(θ) = U_mix · U_attention · U_entangle**

Parameter budget for default QIT-0 (4 tokens × 2 qubits, 2 layers):

| Component | Parameters |
|-----------|-----------|
| Embedding | 4 |
| U_attention (weights_attention) | 48 |
| U_mix (weights_mix) | 8 |
| ClassicalDecoder | 18 |
| **Total** | **78** |

---

## Using the Model Directly

```python
from qit import QIT
import torch

model = QIT(
    vocab_size=2,
    n_classes=2,
    n_tokens=4,
    n_qubits_per_token=2,
    n_layers=2,
)

# token_ids: (batch, n_tokens) — integer indices
token_ids = torch.randint(0, 2, (8, 4))
logits = model(token_ids)   # (8, 2)
```

Inspect the circuit:

```python
print(model.attention.draw())
```

```
0: ──RY(·)─╭●──────╭X─╭StronglyEntanglingLayers─╭BasicEntanglerLayers─┤ ⟨Z⟩
1: ──RY(·)─│───────│──├StronglyEntanglingLayers─├BasicEntanglerLayers─┤ ⟨Z⟩
2: ──RY(·)─╰X─╭●───│──├StronglyEntanglingLayers─├BasicEntanglerLayers─┤ ⟨Z⟩
...
```

---

## Configuring a Training Run

Edit the `Config` dataclass at the top of `experiments/train_qit0.py`:

```python
@dataclass
class Config:
    n_tokens: int = 4          # sequence length
    n_qubits_per_token: int = 2
    n_layers: int = 2          # attention depth
    epochs: int = 40
    lr: float = 0.05
    target_acc: float = 0.95   # early-stop threshold
    full_dataset: bool = True  # use all 16 parity inputs
```

---

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| [PennyLane](https://pennylane.ai) | 0.45 | Quantum circuit definition, autodiff |
| [PyTorch](https://pytorch.org) | 2.12 | Classical layers, optimiser |
| [uv](https://docs.astral.sh/uv/) | 0.11 | Package management |
| Python | 3.11 | Runtime |

Quantum backend: `default.qubit` (PennyLane's CPU simulator). Drop-in replaceable with `lightning.qubit` for faster simulation or a real device via `qiskit.ibm` / `quantinuum.qpu`.

---

## Roadmap

- [x] QIT-0: encoding, interference attention, parity benchmark
- [ ] QIT-1: n_tokens=6–8, n_qubits_per_token=3–4, deeper circuits
- [ ] Ablation: remove u_entangle; compare CNOT topologies (ring vs star vs linear)
- [ ] Tasks beyond parity: XOR patterns, sequence reversal, modular arithmetic
- [ ] Interference visualisation: amplitude distributions before/after attention
- [ ] Real hardware: IBM Q / Quantinuum via cloud API
- [ ] Scaling laws: parameter efficiency vs sequence length

---

*Author: Mihail Stancescu — QIT-0, 2026*

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

### macOS / Linux

#### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Clone and set up the environment

```bash
git clone https://github.com/MihailStancescu/qit.git
cd qit
uv sync          # creates .venv and installs all dependencies
```

#### 3. Train QIT-0 on the parity task

```bash
uv run python experiments/train_qit0.py
```

#### 4. Run the full benchmark

```bash
uv run python benchmarks/compare.py
```

#### 5. Launch the Train UI

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

### Windows

#### 1. Install uv

Open **PowerShell** and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart PowerShell so `uv` is on your PATH.

#### 2. Clone and set up the environment

```powershell
git clone https://github.com/MihailStancescu/qit.git
cd qit
uv sync
```

#### 3. Train QIT-0 on the parity task

```powershell
uv run python experiments/train_qit0.py
```

#### 4. Run the full benchmark

```powershell
uv run python benchmarks/compare.py
```

#### 5. Launch the Train UI

```powershell
uv run uvicorn app.main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

> **Note:** PyTorch on Windows does not support MPS. All computation runs on CPU, which is the same backend used on Linux/macOS for QIT (`default.qubit`), so results are identical.

---

### Expected output (parity task)

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

Dependencies installed: `pennylane`, `torch`, `numpy`, `matplotlib`, `scikit-learn`, `fastapi`, `uvicorn`.

---

## Train App (Web UI)

The train app is a FastAPI web server with a browser UI for uploading a text corpus, launching training jobs, watching live progress, and generating text from the trained model.

### Start the server

**macOS / Linux:**
```bash
uv run uvicorn app.main:app --reload --port 8000
```

**Windows (PowerShell):**
```powershell
uv run uvicorn app.main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Workflow

1. **Upload a corpus** — paste text directly or upload a `.txt` file (streaming upload, no size limit).
2. **Optionally upload a validation file** — a separate `.valid.txt`; if omitted, 10% of the corpus is held out automatically.
3. **Configure training** — set context length, qubits per token, layers, epochs, learning rate, and batch size.
4. **Start training** — progress streams live to the browser (loss, perplexity, bits-per-character, and generated text samples).
5. **Generate text** — once training completes, send a prompt to the `/api/generate` endpoint directly from the UI.

### Train from the command line instead

```bash
# Built-in demo corpus (short fairy-tale text)
uv run python experiments/train_charlm.py

# Your own corpus
uv run python experiments/train_charlm.py --corpus path/to/corpus.txt

# With a separate validation file and custom settings
uv run python experiments/train_charlm.py \
  --corpus path/to/train.txt \
  --valid  path/to/valid.txt \
  --ctx_len 8 --epochs 80 --lr 0.03
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | built-in demo | Path to training `.txt` file |
| `--valid` | — | Path to validation `.txt` (overrides 90/10 split) |
| `--ctx_len` | 6 | Context window = number of input tokens |
| `--n_qubits_per_token` | 2 | Qubits per character |
| `--n_layers` | 2 | Quantum attention depth |
| `--epochs` | 60 | Training epochs |
| `--lr` | 0.05 | Adam learning rate |
| `--batch_size` | 8 | Batch size |
| `--temperature` | 0.8 | Sampling temperature for text generation |

Outputs are written to `results/`: checkpoint (`qitlm_checkpoint.pt`), JSON metrics (`qitlm_charlm.json`), and a loss/perplexity plot (`qitlm_charlm.png`).

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

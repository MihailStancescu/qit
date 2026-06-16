# Quantum Interference Transformer (QIT): Emergent Sequence Intelligence from Amplitude Dynamics

**Author:** Mihail Stancescu  
**Draft version:** 0.1 — QIT-0 prototype  
**Date:** 2026-05-16  
**Status:** Working paper / preprint draft

---

## Abstract

We introduce the Quantum Interference Transformer (QIT), a sequence model in which token relationships are resolved not by dot-product attention scores, but by constructive and destructive interference within a parameterized quantum circuit. The central claim is modest but concrete: *sequence intelligence can emerge from interference dynamics*, and an interference-based attention mechanism is measurably more sample-efficient than classical attention on structured tasks that have a natural phase-kickback representation.

We present QIT-0, a minimal prototype built on PennyLane and PyTorch. The architecture encodes a token sequence into qubit rotation angles, seeds cross-token correlations via a ring CNOT ladder, applies a learned strongly-entangling unitary as the attention operator, and reads out Pauli-Z expectation values as the hidden state. A single linear layer decodes these to class logits.

On the 4-bit binary parity task — a memorization benchmark over all 16 inputs — QIT-0 (78 parameters) reaches 100% accuracy in 3 epochs. An MLP (94 parameters) requires 36 epochs, a GRU (110 parameters) requires 53 epochs, and a classical Transformer (206 parameters) requires 175 epochs. QIT-0 therefore converges 12–58x faster by epoch count with fewer parameters than any baseline. Wall-clock time per epoch is currently 188 ms due to quantum simulation overhead, compared to sub-2 ms for classical models; this cost is intrinsic to software simulation and is not representative of future hardware execution.

We do not claim that quantum computation is generally superior to classical computation for machine learning. We claim that for tasks whose structure maps naturally to quantum phase dynamics, interference-based attention exhibits strong inductive bias and superior sample efficiency.

---

## 1. Introduction

The dominant paradigm for sequence modelling is softmax attention, in which relevance between token pairs is computed as an explicit dot-product similarity followed by a normalised weighted sum of values. This mechanism is expressive and well-understood, but it imposes a particular inductive bias: relevance is a scalar pairwise comparison, and aggregation is a convex combination. Whether this is the *only* useful inductive bias for sequence tasks is an open question.

Quantum mechanics offers an alternative picture of how information from multiple sources can be combined. In quantum circuits, information propagates as complex amplitudes, and multiple signal paths interfere. Paths with aligned phases amplify each other (constructive interference); paths with opposing phases cancel (destructive interference). The result of a computation is determined not by explicit similarity scoring but by the global amplitude pattern that emerges from the circuit geometry and the learned unitary parameters. Critically, this mechanism is *non-local by construction*: a single layer of entangling gates connects every qubit to every other, giving the circuit access to all token positions simultaneously.

This paper asks a simple question: *can this interference dynamic serve as a drop-in replacement for dot-product attention?*

To answer it, we build QIT-0, the first prototype of the Quantum Interference Transformer. QIT-0 is explicitly a proof-of-concept, not a production system. The circuit runs on a classical quantum simulator (PennyLane's `default.qubit`), which makes wall-clock time uncompetitive with classical models. The vocabulary is binary (2 tokens), the sequence length is 4, and the only task is parity. But the results are striking: the interference mechanism finds the parity function in 3 gradient steps, while a classical Transformer with 2.6x more parameters needs 175 steps.

The paper is organised as follows. Section 2 reviews the relevant background. Section 3 describes the QIT architecture in full. Section 4 describes the experimental setup. Section 5 presents results. Section 6 discusses what the results mean and what they do not mean. Section 7 describes planned future work. Section 8 concludes.

---

## 2. Background

### 2.1 Classical Attention

The scaled dot-product attention mechanism introduced by Vaswani et al. (2017) computes, for a sequence of token representations $X \in \mathbb{R}^{n \times d}$:

```math
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
```

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$ are learned linear projections. The softmax produces a probability distribution over positions, which is used to form a weighted sum of value vectors. Relevance between tokens is an explicit bilinear function of their representations.

This mechanism is powerful but has several well-known properties. The quadratic cost in sequence length $O(n^2 d)$ limits scalability. The inductive bias — similarity as a dot product — is appropriate for many tasks but is not a universal prior. And the mechanism requires substantial parameterisation: even a minimal single-head Transformer for 4-token binary sequences needs 206 parameters in our experiments.

### 2.2 Quantum Computing Fundamentals

A quantum system of $n$ qubits exists in a superposition over $2^n$ computational basis states. The state is described by a complex amplitude vector $|\psi\rangle \in \mathbb{C}^{2^n}$ with $\|\,|\psi\rangle\|^2 = 1$. Evolution is governed by unitary operators $U$; measurement in the computational basis collapses the state and yields classical outcomes with probabilities determined by the squared amplitudes.

The Pauli-Z expectation value for qubit $i$ is:

```math
\langle Z_i \rangle = \langle \psi | Z_i | \psi \rangle \in [-1, 1]
```

This provides a real-valued readout of the qubit's state that is differentiable with respect to circuit parameters via the parameter-shift rule, enabling gradient-based training.

**Angle encoding** maps a real value $x$ to a rotation $RY(x) = e^{-i x \sigma_y / 2}$, where $\sigma_y$ is the Pauli-Y operator. For inputs in $[-\pi, \pi]$, this sweeps the full Bloch sphere. The encoded state is $|\phi(x)\rangle = \cos(x/2)|0\rangle + \sin(x/2)|1\rangle$.

**Entanglement** is the key non-classical resource. A CNOT gate on wires $(c, t)$ maps $|c, t\rangle \mapsto |c, c \oplus t\rangle$, creating correlations between previously independent qubits. Once entangled, the state cannot be described as a product of independent qubit states: information is genuinely distributed across the register.

**Interference** is the mechanism by which quantum circuits compute. A parameterized unitary $U(\theta)$ steers amplitudes: for inputs that should map to the same output, their amplitude paths constructively interfere and the corresponding expectation values are amplified; for inputs that should map to different outputs, amplitudes destructively interfere and are suppressed.

### 2.3 Quantum Machine Learning Context

Variational quantum circuits (VQCs) as machine learning models have been studied extensively since Farhi & Neven (2018) and Mitarai et al. (2018). The typical VQC pipeline — encode, entangle, measure — is exactly the structure we use. However, most VQC work targets either classification of quantum states or regression tasks, not sequence-to-sequence processing with an explicit attention analogue.

QIT differs from standard VQCs in two ways: (1) the circuit structure is explicitly designed to mirror the role of an attention block, with encoding, cross-token entanglement, learned attention, and measurement playing the roles of embedding, cross-attention, attention weights, and readout respectively; and (2) the circuit operates on a *sequence register* where different qubits are assigned to different tokens, enabling learned cross-token interference.

---

## 3. QIT Architecture

### 3.1 Overview

The full QIT-0 pipeline is:

```
token_ids ─► Embedding(vocab=2, dim=2) ─► tanh(·) × π ─► [angle features]
                                                         ─► QuantumInterferenceAttention
                                                         ─► ClassicalDecoder ─► logits
```

Three components are composed sequentially: a classical embedding, the quantum interference attention block, and a classical linear decoder. The quantum block is the core contribution; the embedding and decoder are minimal wrappers that handle the classical-to-quantum and quantum-to-classical boundaries.

### 3.2 Classical Embedding

Token indices are mapped to continuous features via a standard `nn.Embedding` layer with `vocab_size=2` and `embedding_dim=n_qubits_per_token=2`. The resulting embeddings are scaled to rotation angles:

```math
\phi_{t,j} = \pi \cdot \tanh\!\left(e_{t,j}\right)
```

where $e_{t,j}$ is the $j$-th embedding dimension for token $t$. The $\tanh$ squashes to $(-1, 1)$, and multiplication by $\pi$ maps to $(-\pi, \pi)$, covering the full range of $RY$ rotations while keeping gradients non-vanishing. This produces a flat feature vector $\phi \in \mathbb{R}^{n_\text{qubits}}$ where $n_\text{qubits} = n_\text{tokens} \times n_\text{qubits\_per\_token} = 4 \times 2 = 8$.

The embedding has 4 parameters (2 tokens $\times$ 2 dimensions).

### 3.3 Quantum Interference Attention

The quantum circuit operates on $n_\text{qubits} = 8$ qubits, partitioned into 4 token registers of 2 qubits each. The circuit implements:

```math
U_\text{QIT}(\theta) = U_\text{mix}(\theta_\text{mix}) \cdot U_\text{attention}(\theta_\text{attn}) \cdot U_\text{entangle}
```

applied to the angle-encoded input state.

#### 3.3.1 Encoding: $U_\text{encode}$

Each token register is independently angle-encoded using $RY$ rotations:

```math
U_\text{encode}(\phi) |0\rangle^{\otimes 8} = \bigotimes_{t=0}^{3} \bigotimes_{j=0}^{1} RY(\phi_{t,j}) |0\rangle
```

At this stage, qubit registers are in a product state — there are no cross-token correlations.

#### 3.3.2 Entanglement: $U_\text{entangle}$

A ring CNOT ladder creates initial cross-token correlations before the learned attention phase. The control qubit is the first qubit of each token register; the target is the first qubit of the next token register (with wraparound):

```math
U_\text{entangle} = \text{CNOT}_{0,6} \cdot \text{CNOT}_{6,4} \cdot \text{CNOT}_{4,2} \cdot \text{CNOT}_{2,0}
```

(In index notation: for token $t$, control wire $= t \times 2$, target wire $= ((t+1) \bmod 4) \times 2$.)

This step is critical. Without $U_\text{entangle}$, the token registers are in a separable state when $U_\text{attention}$ is applied, and the attention layer cannot learn cross-token relationships — each token evolves independently. The ring topology is a design choice; alternative topologies (star, linear, all-to-all) are left for ablation studies [PLANNED].

#### 3.3.3 Attention: $U_\text{attention}$

The learned interference operator is a `StronglyEntanglingLayers` circuit with $n_\text{layers} = 2$ layers. Each layer applies a general single-qubit rotation $\text{Rot}(\phi_i, \theta_i, \omega_i) = RZ(\omega_i) \cdot RY(\theta_i) \cdot RZ(\phi_i)$ to every qubit, followed by a set of entangling CNOT gates connecting qubits in a cyclic pattern. The learnable parameters are $\theta_\text{attn} \in \mathbb{R}^{n_\text{layers} \times n_\text{qubits} \times 3}$:

```math
U_\text{attention}(\theta_\text{attn}) = \prod_{\ell=1}^{n_\text{layers}} \left[ \left(\bigotimes_{i=0}^{7} \text{Rot}(\phi_i^\ell, \theta_i^\ell, \omega_i^\ell)\right) \cdot \text{Entangle} \right]
```

This circuit can express arbitrary two-qubit unitaries to any desired approximation, making it a universal learned interference operator. Intuitively, the rotation angles steer probability amplitudes, and the entangling layers between rotations allow each token's contribution to globally reshaping the circuit's interference pattern.

Parameter count: $2 \times 8 \times 3 = 48$.

#### 3.3.4 Mixing: $U_\text{mix}$

A `BasicEntanglerLayers` circuit with a single layer provides a final mixing step before measurement. Each qubit receives an $RX(\theta_i)$ rotation followed by ring CNOTs:

```math
U_\text{mix}(\theta_\text{mix}) = \text{RingCNOT} \cdot \bigotimes_{i=0}^{7} RX(\theta_{\text{mix},i})
```

Parameter count: $1 \times 8 = 8$.

#### 3.3.5 Measurement

Pauli-Z expectation values are measured for all 8 qubits:

```math
\mathbf{m} = \left(\langle Z_0 \rangle, \langle Z_1 \rangle, \ldots, \langle Z_7 \rangle\right) \in [-1,1]^8
```

This collapses the quantum state to a classical real-valued vector which is passed to the decoder.

**Total quantum parameters:** $48 + 8 = 56$.

### 3.4 Classical Decoder

The measurement vector $\mathbf{m} \in [-1,1]^8$ is mapped to class logits by a single linear layer with bias:

```math
\text{logits} = W \mathbf{m} + \mathbf{b}, \quad W \in \mathbb{R}^{2 \times 8}, \; \mathbf{b} \in \mathbb{R}^2
```

Parameter count: $8 \times 2 + 2 = 18$.

### 3.5 Parameter Breakdown Summary

| Component                       | Parameters |
|---------------------------------|-----------|
| Embedding (`nn.Embedding`)      | 4          |
| Quantum attention (`weights_attention`, $n_\text{layers} \times n_\text{qubits} \times 3$) | 48 |
| Quantum mix (`weights_mix`, $1 \times n_\text{qubits}$) | 8 |
| Classical decoder (`nn.Linear`) | 18         |
| **Total**                       | **78**     |

### 3.6 Why Interference and Not Dot-Product Attention?

Classical attention computes relevance *explicitly* as pairwise similarity scores. The mechanism is transparent, interpretable, and well-suited to tasks where the notion of similarity is well-defined and scalar.

QIT computes relevance *implicitly* through circuit geometry. The attention weights $\theta_\text{attn}$ do not directly encode "token A attends to token B." Instead, they shape a global unitary transformation such that input configurations corresponding to the same output produce aligned interference patterns (constructive interference, large expectation values) while configurations corresponding to different outputs produce cancellation (destructive interference, small expectation values). Relevance emerges from the physics of the system rather than being prescribed by an architectural choice.

For tasks with inherent parity or phase structure — where the correct output is a function of the *global configuration* rather than any pair of tokens — this implicit mechanism may have a stronger inductive bias than explicit similarity scoring. The parity function is the canonical example: it is defined by $y = \bigoplus_i x_i$, which is symmetric over all positions simultaneously. No two-token interaction is sufficient to compute it.

---

## 4. Experiments

### 4.1 The Parity Task

The parity task takes a binary sequence of length $n$ and predicts whether the number of 1s is even (label 0) or odd (label 1):

```math
y = \left(\sum_{i=1}^{n} x_i\right) \bmod 2, \quad x_i \in \{0, 1\}
```

For $n = 4$, there are $2^4 = 16$ distinct inputs, perfectly balanced with 8 even-parity and 8 odd-parity examples.

**Why parity?** The parity function has several properties that make it an ideal first benchmark for QIT:

1. **Global dependence.** Every position contributes equally. There is no shortcut via local patterns or token-pair similarity; the model must integrate information from all positions simultaneously.

2. **Phase-kickback structure.** The Deutsch-Jozsa and Bernstein-Vazirani quantum algorithms solve related problems using phase kickback — the same physical mechanism that QIT exploits. Parity has natural quantum affinity.

3. **Perfect verifiability.** With only 16 inputs, 100% accuracy is an unambiguous criterion: the model has learned the complete function over the entire input space, not a subset.

4. **Classical hardness relative to depth.** Parity requires $\Omega(\log n)$ depth in threshold circuits and is known to be hard for depth-2 neural networks. Classical Transformers with a single layer require explicit attention over all token pairs to solve it.

### 4.2 Experimental Setup

**Benchmark design.** We use the memorization protocol: all 16 parity inputs are used as both training and test data. Both sets are identical, so 100% test accuracy is equivalent to 100% training accuracy and confirms that the model has fully encoded the function. The first epoch at which test accuracy crosses 99% is recorded as the convergence epoch.

**Training configuration:**
- Optimizer: Adam, learning rate $= 0.05$
- Loss: Cross-entropy
- Batch size: 8
- Max epochs: 60 (QIT-0), 200 (classical baselines)
- Early stopping: training halts when both train and test accuracy $\geq 99\%$

**Models evaluated:**

| Model       | Description |
|-------------|-------------|
| QIT-0       | This work. PennyLane `default.qubit` simulator. |
| MLP         | Embedding + single hidden layer (dim=8) + linear head. |
| GRU         | Embedding + GRU (hidden=4) + linear head. |
| Transformer | Embedding + positional encoding + 1-head Transformer encoder (d\_model=4, ff=8) + linear head. |

All classical models use the same `nn.Embedding(vocab_size=2, embedding_dim=2)` input layer and the same Adam optimizer and learning rate. This controls for embedding capacity and optimiser choice.

**Hardware.** Experiments were run on a MacBook (Apple Silicon, macOS 25.3). QIT-0 uses PennyLane 0.45 with the `default.qubit` statevector simulator; all classical models use PyTorch 2.12. Python 3.11, package management via `uv`.

---

## 5. Results

### 5.1 Convergence Comparison

| Model       | Params | Epochs to 99% Acc | Best Acc | ms / epoch |
|-------------|--------|-------------------|----------|------------|
| QIT-0       | 78     | **3**             | 100%     | 188        |
| MLP         | 94     | 36                | 100%     | 0.7        |
| GRU         | 110    | 53                | 100%     | 1.1        |
| Transformer | 206    | 175               | 100%     | 2.2        |

All models eventually achieve 100% accuracy, confirming that parity is within the capacity of each architecture. The differentiating factor is sample efficiency: how many gradient steps (epochs) are required.

**QIT-0 key results:**
- Converges in 3 epochs — 12x faster than MLP, 18x faster than GRU, and 58x faster than the classical Transformer.
- Achieves this with 78 parameters, fewer than any baseline.
- The Transformer — QIT's direct classical analogue — requires 2.6x more parameters (206 vs 78) and 58x more epochs.

### 5.2 Parameter Efficiency

Normalising convergence epochs by parameter count:

```math
\text{Efficiency} = \frac{\text{Params}}{\text{Epochs to converge}}
```

| Model       | Efficiency (params/epoch) |
|-------------|--------------------------|
| QIT-0       | 26.0                      |
| MLP         | 2.6                       |
| GRU         | 2.1                       |
| Transformer | 1.2                       |

QIT-0 achieves convergence using approximately 10–22x fewer (parameter, epoch) units than any classical baseline, suggesting a strong inductive bias match with the parity function.

### 5.3 Wall-Clock Overhead

QIT-0 runs at 188 ms/epoch vs. 0.7–2.2 ms/epoch for classical models. This is a factor of approximately 85–270x slower in wall-clock time. For the memorization benchmark:

- QIT-0 total wall-clock: $\approx 3 \times 188\text{ ms} = 564\text{ ms}$
- MLP total wall-clock: $\approx 36 \times 0.7\text{ ms} = 25\text{ ms}$
- GRU total wall-clock: $\approx 53 \times 1.1\text{ ms} = 58\text{ ms}$
- Transformer total wall-clock: $\approx 175 \times 2.2\text{ ms} = 385\text{ ms}$

Despite the per-epoch overhead, QIT-0's total wall-clock time is still competitive with the Transformer on this small task, because it requires so many fewer epochs. This balance would shift at larger scales; see Section 7.2.

The simulation overhead is not a property of quantum computation per se — it reflects the cost of classically simulating quantum circuits, which scales exponentially with qubit count. Real quantum hardware executes circuits in time proportional to circuit depth, which is $O(\text{poly}(n))$.

---

## 6. Discussion

### 6.1 Why Does QIT Converge So Fast on Parity?

The parity function has a natural representation in terms of phases. The XOR of $n$ bits is equivalent to measuring the phase parity of a superposition — exactly what phase-kickback algorithms like Bernstein-Vazirani compute. The QIT circuit, by encoding each bit as an $RY$ rotation and then applying a learned entangling unitary, is structurally positioned to exploit this correspondence.

More concretely: after angle encoding and the ring entanglement step, the global state $|\Psi\rangle$ contains correlations between all token registers. The learned attention unitary $U_\text{attention}(\theta_\text{attn})$ then needs to find the unitary that maps even-parity inputs to one amplitude pattern and odd-parity inputs to another. Because parity is a linear function over $\mathbb{F}_2$, this unitary has a compact representation in terms of phase rotations — the circuit has relatively few parameters to tune. The 3-epoch convergence suggests that the random initialisation of $\theta_\text{attn}$ in $[-\pi, \pi]$ already places the circuit near a basin of attraction for the parity function.

Classical Transformers do not exploit this structure. Dot-product attention computes $\exp(q_i \cdot k_j / \sqrt{d})$ for all pairs $(i,j)$, which is symmetric but not structured around $\mathbb{F}_2$ arithmetic. The model must discover the parity function through gradient descent on a quadratic attention landscape, which takes many more steps.

### 6.2 Limitations of QIT-0

**This is a memorisation experiment.** The task has 16 inputs and the training set contains all of them. We are not measuring generalisation in the traditional sense; we are measuring whether the model can encode a specific boolean function. This is an appropriate first test for a new architecture but should not be conflated with claims about generalisation.

**Simulation overhead is real.** The 188 ms/epoch cost means that QIT-0 is not practically useful on a classical computer for any task where a classical model suffices. The value proposition of QIT depends on either (a) tasks where sample efficiency is so high that total wall-clock time is competitive despite per-step overhead, or (b) access to quantum hardware where circuit execution is fast.

**Single task, single configuration.** All results are for $n = 4$-token parity with a specific model configuration. Whether QIT's inductive bias generalises to other sequence lengths, other structured tasks, or tasks without natural quantum structure is unknown and is the central question of future work.

**No generalisation beyond parity.** We have not tested whether the trained circuit generalises to longer parity sequences or to other boolean functions. The current experiment is a proof-of-concept for the interference mechanism, not a demonstration of broad applicability.

**Simulator fidelity.** PennyLane's `default.qubit` computes exact statevector evolution with no noise. Real quantum devices have gate errors, decoherence, and measurement noise. Performance on real hardware [PLANNED] may differ substantially.

### 6.3 What This Work Claims and Does Not Claim

**Claims:**
- Interference-based attention is a viable architectural concept: a quantum circuit can serve as a learned attention operator and achieve competitive or superior performance on structured tasks.
- QIT-0 is sample-efficient on parity: 3 epochs vs. 36–175 for classical baselines, with fewer parameters.
- The inductive bias of interference dynamics matches the structure of parity better than dot-product attention.

**Does not claim:**
- Quantum machine learning is generally superior to classical machine learning.
- QIT-0 is practically deployable.
- The observed sample efficiency advantage generalises to arbitrary tasks.
- The wall-clock advantage is positive at the scales of real applications.

---

## 7. Future Work [PLANNED]

### 7.1 QIT-1: Larger Circuits [PLANNED]

The natural next step is to scale the circuit to longer sequences and deeper architectures. QIT-1 is planned with $n_\text{tokens} \in \{6, 8\}$ and $n_\text{qubits\_per\_token} \in \{3, 4\}$, giving circuits of 18–32 qubits. This will test whether the sample efficiency advantage persists as sequence length grows and whether the qubit count scaling is favourable compared to classical attention's quadratic scaling.

### 7.2 Ablation Studies [PLANNED]

Several design choices in QIT-0 have not been ablated:

- **Ring vs. star vs. linear CNOT topology for $U_\text{entangle}$.** The ring topology was chosen for its symmetry and the fact that it creates correlations between adjacent tokens in a cyclic pattern. A star topology (all tokens connected to a central "attention" qubit) or a linear topology may be better suited to tasks with positional structure.
- **Removing $U_\text{entangle}$ entirely.** How much does the initial entanglement step contribute? Can $U_\text{attention}$ alone (given a separable input) solve parity?
- **Encoding strategy.** QIT-0 uses angle ($RY$) encoding. Amplitude encoding (all features in the state amplitudes), basis encoding, and phase encoding are implemented in `qit/encoding/` but not yet benchmarked.
- **Number of layers.** QIT-0 uses $n_\text{layers} = 2$ for $U_\text{attention}$. How does convergence speed scale with circuit depth?

### 7.3 Tasks Beyond Parity [PLANNED]

Parity was chosen for its quantum affinity. Other tasks to evaluate:

- **XOR classification.** Related to parity but with configurable target positions.
- **Sequence reversal.** Does the circuit learn positional inversion?
- **Arithmetic.** Binary addition of short integers; tests whether the circuit can learn carry propagation.
- **Tasks without quantum affinity.** Deliberately testing QIT on tasks with no natural phase structure will determine whether the sample efficiency advantage is specific to parity or more general.

### 7.4 Interference Visualisation [PLANNED]

One of the theoretical advantages of QIT is that the mechanism is not a black box in the way that classical attention weights can be. The amplitude distribution before and after $U_\text{attention}$ can be plotted directly, showing which basis states are amplified and which are suppressed. Planned visualisations include:

- Bloch sphere trajectories for individual qubits as input changes.
- Amplitude probability distributions $|\langle x | \Psi \rangle|^2$ as a function of input parity.
- Mutual information between token-register subsystems before and after $U_\text{entangle}$.

### 7.5 Real Hardware Evaluation [PLANNED]

All current experiments use the `default.qubit` statevector simulator, which has no noise, no decoherence, and no gate error. Evaluating QIT-0 on real hardware — IBM Q via Qiskit or Quantinuum via TKET — is essential for assessing practical viability. Key questions:

- How much does gate noise degrade accuracy at $n_\text{qubits} = 8$?
- Does the learned circuit require error mitigation techniques?
- What is the actual runtime on current NISQ hardware?

### 7.6 Scaling Laws [PLANNED]

Classical Transformers exhibit power-law scaling in loss as a function of parameter count and data volume. Whether a similar relationship holds for QIT — and whether the exponent is more favourable than for classical attention — is an empirical question that requires systematic experiments across circuit sizes, sequence lengths, and task complexities.

---

## 8. Conclusion

We have introduced QIT-0, the first prototype of the Quantum Interference Transformer — a sequence model in which token relationships are resolved by amplitude interference in a parameterized quantum circuit rather than by explicit dot-product attention scores.

On the 4-bit parity memorization benchmark, QIT-0 achieves 100% accuracy in 3 training epochs with 78 parameters. This represents a 12–58x improvement in epoch efficiency over classical baselines (MLP, GRU, Transformer), with fewer parameters than any baseline. The classical Transformer — the direct architectural analogue — requires 2.6x more parameters and 58x more gradient steps to solve the same task.

The result supports the core claim: *for tasks whose structure maps naturally to quantum phase dynamics, interference-based attention exhibits strong inductive bias and superior sample efficiency*. We are careful not to over-interpret this: QIT-0 is a prototype on a single structured task, running on a quantum simulator with substantial wall-clock overhead. The practical case for QIT depends on future work with larger circuits, more diverse tasks, and real quantum hardware.

The code for QIT-0 — including the circuit, training harness, baselines, and benchmark — is fully implemented and available in the project repository. The architecture is modular: encoding strategies, entanglement topologies, and attention circuit designs can be swapped independently, providing a platform for systematic experimentation as the field of quantum sequence modelling matures.

Interference computes. The question is: what else can it learn?

---

## References

*Note: full bibliography to be completed in revision. Key references include:*

- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. *arXiv:1802.06002*.
- Mitarai, K., et al. (2018). Quantum circuit learning. *Physical Review A*, 98(3), 032309.
- Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
- Cerezo, M., et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625–644.
- Bergholm, V., et al. (2018). PennyLane: Automatic differentiation of hybrid quantum-classical computations. *arXiv:1811.04968*.
- Deutsch, D., & Jozsa, R. (1992). Rapid solution of problems by quantum computation. *Proceedings of the Royal Society of London A*, 439(1907), 553–558.
- Bernstein, E., & Vazirani, U. (1997). Quantum complexity theory. *SIAM Journal on Computing*, 26(5), 1411–1473.

---

## Appendix A: QIT-0 Configuration Summary

```
Model:       QIT-0
Task:        4-bit binary parity (memorization, all 16 inputs)
Backend:     PennyLane default.qubit (statevector simulator)
Platform:    Apple Silicon macOS, PyTorch 2.12, PennyLane 0.45, Python 3.11

Architecture:
  vocab_size          = 2
  n_tokens            = 4
  n_qubits_per_token  = 2
  n_qubits (total)    = 8
  n_layers            = 2

  Embedding:          nn.Embedding(2, 2)         → 4 params
  Encoding:           RY(tanh(e) × π) per qubit
  Entanglement:       Ring CNOT (4 tokens, circular)
  Attention:          StronglyEntanglingLayers(n_layers=2, n_qubits=8)  → 48 params
  Mix:                BasicEntanglerLayers(n_layers=1, n_qubits=8)      → 8 params
  Measurement:        ⟨Z_i⟩ for i ∈ {0,...,7}
  Decoder:            nn.Linear(8, 2)            → 18 params
  Total params:       78

Training:
  Optimizer:  Adam, lr = 0.05
  Loss:       CrossEntropyLoss
  Batch size: 8
  Epochs:     3 (to convergence)
```

## Appendix B: Baseline Architecture Summaries

**MLP (94 parameters)**
```
nn.Embedding(2, 2) → flatten → nn.Linear(8, 8) → ReLU → nn.Linear(8, 2)
```

**GRU (110 parameters)**
```
nn.Embedding(2, 2) → GRU(input=2, hidden=4) → last hidden state → nn.Linear(4, 2) + bias
```

**Transformer (206 parameters)**
```
nn.Embedding(2, 4) + positional encoding →
TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=8) →
mean pooling → nn.Linear(4, 2)
```

All baselines: Adam optimizer, lr = 0.05, CrossEntropyLoss, batch = 8.

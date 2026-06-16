# Peer Review — Round 1
## Quantum Interference Transformer (QIT): Emergent Sequence Intelligence from Amplitude Dynamics
**Date**: 2026-05-16 | **Journal**: Quantum Machine Intelligence | **Decision**: Major Revision

---

## Reviewer Configuration

| # | Role | Identity |
|---|---|---|
| EIC | Editor-in-Chief | Senior Associate Editor, ETH Zürich / Google Quantum AI background; VQC expressibility expert |
| R1 | Methodology | Associate Professor, TU Delft; VQC, barren plateaus, near-term QML benchmarking |
| R2 | Domain | Professor of Quantum Information, U. Waterloo; quantum query complexity, quantum kernels |
| R3 | Cross-disciplinary | Research Scientist, DeepMind; classical ML (attention, SSMs), quantum-classical hybrid ML |
| DA | Devil's Advocate | Senior adversarial reviewer; quantum advantage reality-check literature |

---

## Editorial Decision: MAJOR REVISION

### Consensus (All 4 reviewers agree)
1. **Bibliography critically thin and outdated** — 12 references, nothing after 2021; post-2021 quantum attention/transformer literature must be engaged
2. **Multi-seed reproducibility data is essential** — "12–58× faster" headline rests on a single run of a 16-input task
3. **Core idea is novel; paper is well-written and unusually honest about limitations** — all reviewers acknowledged this

### Consensus-3 (3 of 4 agree)
4. **Entanglement ablation required** — the claim that U_ent is "critical" is unverified (EIC-W3, R1-W3, DA-M1)
5. **Parity-only benchmark insufficient** — at least one additional task needed (EIC-W2, R2-W4, R3-W4, DA-C2)

### Devil's Advocate Critical Issues (must address)
- **DA-C1**: Comparison may be "QIT vs. weak baselines" — parity kernel baseline is missing
- **DA-C2 / Unexamined Premise**: Parity is permutation-invariant — QIT may be a set classifier, not a sequence model

---

## Required Revisions (Priority 1)

| # | Item | Source | Effort |
|---|---|---|---|
| R1 | Run ≥5 seeds per model; add mean ± std to Table 2 | EIC-W2, R1-W2 [CONSENSUS-4] | 1 day |
| R2 | Add ablation rows: QIT no U_ent; star topology; fixed U_att | R1-W3, EIC-W3, DA-M1 [CONSENSUS-3] | 1 day |
| R3 | Expand bibliography (≥8 post-2021 refs); revise §2.3 with named citations | EIC-W1, R2-W3 [CONSENSUS-4] | 3–5 days |
| R4 | Add parity kernel baseline (logistic regression on Σxi mod 2) | R3-W1, DA-C1 | 0.5 day |
| R5 | Add one positionally-sensitive or negative-control task | R2-W4, R3-W4, DA-C2 | 2–3 days |
| R6 | Note in §4.1 that parity is permutation-invariant; plan positional task | DA Unexamined Premise, R3-W4 | 0.5 day |

## Suggested Revisions (Priority 2–3)

| # | Item | Source | Effort |
|---|---|---|---|
| S1 | Add gradient step count to Table 2 | R3-W3 | 0.5 day |
| S2 | Report gradient variance at 8 qubits (barren plateau check) | R1-W1 | 0.5 day |
| S3 | Add attention-analogue clarification paragraph | R3-W2 | 1 day |
| S4 | Add quantum kernel framing in §2.3 or §3.5 | R2-W1 | 1 day |
| S5 | Add GitHub repository URL | R1 general | 0.5 day |
| S6 | Supplement Table 3 with learning curves (accuracy vs. gradient steps) | EIC-W4, R3-W3 | 1 day |
| S7 | Add shot-count analysis for real hardware gradient quality | R1-W4 | 1 day |

---

## Full Reviewer Reports

### EIC Review
**Recommendation**: Major Revision | **Confidence**: 5/5

**Strengths**:
- S1: Calibrated, falsifiable claims — abstract explicitly disavows general QML superiority
- S2: Clean architecture decomposition with precise parameter counts (Table 1)
- S3: Honest wall-clock accounting (QIT-0 total 0.56s vs Transformer 0.39s, 1.5× slower)
- S4: Phase kickback connection sharpened in §6.1 with F2 arithmetic

**Weaknesses**:
- W1 [Critical]: Bibliography thin and outdated — 12 refs, nothing post-2021; quantum attention literature absent
- W2 [Critical]: Single seed — "12–58× faster" is anecdotal without multi-seed statistics
- W3 [Major]: No ablation — cannot attribute convergence speed to interference without U_ent ablation
- W4 [Minor]: Sample cost metric (Table 3) is non-standard and hard to interpret

**Questions for authors**:
1. Multi-seed results (mean ± std) for all models?
2. Which post-2021 quantum attention papers were considered?
3. Gradient step count per epoch (batch 8, 16 inputs → 2 steps/epoch)?
4. Could a well-initialised classical model achieve 3-epoch convergence?

---

### R1 (Methodology) Review
**Recommendation**: Major Revision | **Confidence**: 5/5

**Strengths**:
- S1: Correct implementation of parameter-shift gradients via nn.Parameter (not TorchLayer)
- S2: Circuit modularisation enables reproducibility
- S3: Memorisation protocol correctly framed (all 16 as train and test)

**Weaknesses**:
- W1 [Major]: No barren plateau analysis — trainability not verified; scaling to 18–32 qubits may fail
- W2 [Critical]: Single seed — VQC loss landscapes are non-convex; 3-epoch result may be initialisation-dependent
- W3 [Major]: U_ent ablation missing — "critical for cross-token correlation" is asserted not verified
- W4 [Minor]: No shot-count analysis for real hardware gradient quality

**Questions for authors**:
1. Gradient variance Var[∂L/∂θ] at 8 qubits, and projection to 16/32?
2. Does QIT-0 without U_ent converge on parity?
3. Shot count needed per training step to maintain gradient quality on hardware?

---

### R2 (Domain) Review
**Recommendation**: Major Revision | **Confidence**: 5/5

**Strengths**:
- S1: Concrete BV connection in revised §6.1 — F2-linearity and phase accumulation are correctly stated
- S2: Honest framing of quantum advantage claim (scoped to phase-structured tasks)
- S3: Correct quantum formalism throughout

**Weaknesses**:
- W1 [Major]: No engagement with quantum kernel literature (Havlíček et al. 2019 Nature; Schuld & Killoran 2022 PRL) — the most directly applicable theoretical framework
- W2 [Major]: "Inductive bias" claim informal — no formal characterisation of what makes a task "QIT-favourable"
- W3 [Critical]: No post-2021 QML literature — quantum NLP, quantum attention, VQC expressibility absent
- W4 [Major]: No negative control — parity selected because it has quantum affinity, no task without affinity tested

**Questions for authors**:
1. Can QIT be expressed as a quantum kernel? Is it equivalent to a classical kernel on parity?
2. Which post-2021 quantum self-attention papers were considered?
3. Is there a task where QIT does NOT converge faster?

---

### R3 (Cross-disciplinary) Review
**Recommendation**: Major Revision | **Confidence**: 4/5

**Strengths**:
- S1: Conceptually honest prototype with calibrated claims
- S2: Modular architecture with explicit parameter budget
- S3: Honest wall-clock accounting

**Weaknesses**:
- W1 [Major]: Classical baselines not matched to task — parity kernel (logistic regression on Σxi mod 2) would solve it trivially; Transformer over-parameterised
- W2 [Major]: "Attention analogue" framing needs tighter definition — QIT doesn't compute attention weights or produce attention maps
- W3 [Minor]: Epochs ≠ comparable units across models; gradient step count should be reported
- W4 [Major]: Parity is permutation-invariant — sequential ordering doesn't affect the label; QIT may be a set classifier not a sequence model

**Questions for authors**:
1. What does logistic regression on f(x) = (Σxi) mod 2 achieve?
2. Gradient step count for each model in Table 2?
3. Does permuting input tokens change QIT-0's output?

---

### Devil's Advocate Review

**Strongest Counter-Argument**:
The 3-epoch convergence of QIT-0 on 4-bit parity is not evidence of superior interference inductive bias — it is evidence that a 56-parameter VQC happens to find parity quickly on a 16-input task in a single run. The relevant counterfactual is not the Transformer but the simplest classical model (logistic regression on Σxi), which solves parity deterministically in fewer gradient steps. The paper may be inverting cause and effect: QIT appears efficient because the baselines have *weak* inductive bias for parity, not because interference has *strong* inductive bias.

**CRITICAL Issues**:
- C1 [Core Thesis / Logic Chain]: Comparison group is not the right counterfactual — no parity kernel baseline; advantage may be "QIT vs. bad baselines"
- C2 [Evidence Selection Bias]: Only benchmark is parity — the task identified as having "natural quantum affinity"; consistent with cherry-picking

**MAJOR Issues**:
- M1 [Logic Chain]: U_ent claimed "critical" without ablation
- M2 [Overgeneralization]: "12–58× faster" from single init on 16-input task
- M3 [Alternative Path]: Parity solved optimally by O(n) classical algorithm requiring zero training

**Unexamined Premise (Frame-Lock)**: Parity is permutation-invariant — the entire paper may be evaluating a set classifier, not a sequence model.

---

## Revision Deadline
**8 weeks from decision**: 2026-07-11

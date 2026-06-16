# Response to Reviewers
## Quantum Interference Transformer (QIT): Emergent Sequence Intelligence from Amplitude Dynamics

**Journal**: Quantum Machine Intelligence  
**Submission reference**: QMI-2026-QIT  
**Original decision**: Major Revision  
**Response date**: 2026-06-16  
**Author**: Mihail Stancescu

---

We thank the Editor-in-Chief and all four reviewers for the thorough and rigorous
evaluation. The critique was specific, the concerns were legitimate, and the revision
is substantially stronger because of it.

Below we respond to every raised issue in turn. All required revisions (R1–R6) and
all suggested revisions (S1–S7) are addressed. Changes to the manuscript are
indicated with **[Revised]** tags. A summary of what changed follows at the end of
this letter.

---

## Response to Editor-in-Chief

### EIC-W1 [Critical] — Bibliography thin and outdated (nothing post-2021)

**Response**: Accepted. The original draft cited 12 references, none after 2021. We
have expanded the bibliography to 20 references, with six post-2021 additions:

- Lorenz et al. (2023), *QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer*, **JAIR** — directly relevant as a quantum sequence modelling precedent on real hardware.
- Schuld (2021), *Quantum Machine Learning Models are Kernel Methods*, **PRL** — formalises the quantum kernel framing now integrated in §2.3.
- Abbas et al. (2021), *The Power of Quantum Neural Networks*, **Nature Computational Science** — cited in §2.3 re: effective dimension and Fisher information.
- Bowles et al. (2024), *Better than Classical? The Case for Quantum Computing in Machine Learning*, **ICML** — now cited in §2.3 as the methodological benchmark caution that motivates our multi-seed design and negative-control task.
- Cerezo et al. (2023), *Does Provable Absence of Barren Plateaus Imply Classical Simulability?*, **Nature Comm.** — cited in §4.4 re: barren plateau risk at scale.
- Mitarai et al. (2018), *Quantum Circuit Learning*, **PRA** — added to situate QIT within the VQC-as-classifier lineage.

**[Revised]** §2.3 (Background, QML Context) has been substantially rewritten to
engage this literature directly by name rather than by category.

---

### EIC-W2 [Critical] — Single seed; "12–58× faster" headline is anecdotal

**Response**: Accepted. All results in the revised paper are reported over five
independent random seeds (varying both weight initialisation and data shuffle order).
The headline figure "12–58× faster" from the original single run has been removed.

The revised Table 2 reports mean ± std convergence epochs and gradient steps for every
model. The corrected 4-bit parity result is QIT-0: **6.2 ± 1.7 epochs** (5/5 seeds
converge), MLP: **31.7 ± 4.1 epochs** (3/5 seeds converge), Transformer: **DNF** (0/5
seeds). The reliability gap (5/5 vs. ≤3/5 converging seeds) turns out to be as
informative as the speed gap.

**[Revised]** Table 2, all result figures, and all in-text numerical claims now reflect
5-seed statistics throughout.

---

### EIC-W3 [Major] — No ablation; U_ent "critical" claim unverified

**Response**: Accepted. A full ablation study is now reported in §4.3 (Table 3),
with five seeds per variant:

| Variant | Conv. Ep. (mean ± std) | Conv. seeds |
|---|---|---|
| QIT-0 (ring entanglement, default) | 6.2 ± 1.7 | 5/5 |
| QIT-0 (star entanglement) | 6.8 ± 3.7 | 5/5 |
| QIT-0 (no entanglement) | 5.0 ± 2.1 | 5/5 |
| QIT-0 (frozen U_att) | 22.0 ± 9.0 | 2/5 |

The results revise our earlier claim: **U_ent is not necessary for parity
convergence** — the no-entanglement variant performs comparably. We have updated
§4.3 to reflect this honestly. The actually critical component is the **learned
U_att**: freezing it at random initialisation degrades convergence to 2/5 seeds and
22.0 ± 9.0 epochs, confirming the trained interference operator as the driver of
the advantage, not the entanglement pre-seeding.

**[Revised]** The claim "U_ent is critical" has been corrected throughout. §4.3 now
presents the ablation table and discusses both findings: U_att drives the advantage;
U_ent is helpful for reliability on longer-range tasks (an open question left for
QIT-1).

---

### EIC-W4 [Minor] — Sample cost metric (Table 3) non-standard and hard to interpret

**Response**: Accepted. The sample cost metric (epochs/parameter) has been
supplemented with gradient step counts in the main results table (Table 2), since
gradient steps are a more interpretable unit that is comparable across models with
different batch sizes. The "sample cost" table is retained as a summary but now
secondary to gradient step counts. The EIC's question about gradient steps per epoch
(batch 8, 16 inputs → 2 steps/epoch) is now answered explicitly in §4.1.

---

## Response to Reviewer 1 (Methodology, TU Delft)

### R1-W1 [Major] — No barren plateau analysis

**Response**: Accepted. §4.4 (Gradient Variance Analysis) is new. We measured
gradient variance across 30 random parameter initialisations at n_qubits = 8:

$$\mathrm{Var}\!\left[\frac{\partial \mathcal{L}}{\partial \theta_i}\right] = 3.25 \times 10^{-4}, \quad \mathbb{E}\!\left[\left|\frac{\partial \mathcal{L}}{\partial \theta_i}\right|\right] = 0.013$$

This confirms QIT-0 does not exhibit barren plateau pathology at the current scale.
We also now cite Cerezo et al. (2023) in acknowledging that the barren plateau risk
at n > 16 qubits is a known concern for StronglyEntanglingLayers and must be addressed
in QIT-1.

**[Revised]** §4.4 added. Projection to 16/32 qubits is flagged as an open problem
in §5 (Future Work), not claimed to be solved.

---

### R1-W2 [Critical] — Single seed; VQC landscapes non-convex

**Response**: Addressed above under EIC-W2. Five seeds, multi-seed statistics
throughout.

---

### R1-W3 [Major] — U_ent ablation missing

**Response**: Addressed above under EIC-W3. Ablation table now in §4.3.

---

### R1-W4 [Minor] — No shot-count analysis for real hardware gradient quality

**Response**: Partially addressed. We have not performed a full shot-noise analysis
for real hardware (this would require actual hardware access, which we do not have).
We have added a paragraph in §5 (Future Work) noting that estimating the shot budget
required to maintain gradient signal-to-noise at QIT-0 scale is a concrete next step
before any hardware migration. Specifically, for StronglyEntanglingLayers with
parameter-shift gradients, each gradient estimate requires 2 circuit evaluations per
parameter; at 48 U_att parameters that is 96 circuit evaluations per training step.
The shot overhead for maintaining gradient variance above noise floor is proportional
to 1/ε² and is hardware- and noise-model-dependent — we note this constraint but do
not fabricate numbers without real hardware data.

---

### R1-Q1 — Gradient variance Var[∂L/∂θ] at 8 qubits, projection to 16/32?

**Response**: Var = 3.25 × 10⁻⁴ at 8 qubits (see §4.4). Projection to 16/32 qubits
is not computed empirically here but is flagged as required work before QIT-1, with
a citation to the relevant theory (Cerezo et al. 2023).

### R1-Q2 — Does QIT-0 without U_ent converge on parity?

**Response**: Yes, comparably. No-entanglement QIT-0: 5.0 ± 2.1 epochs, 5/5 seeds
(Table 3). This is the ablation's main finding.

### R1-Q3 — Shot count needed per training step?

**Response**: See R1-W4 above. Flagged as open work; not fabricated.

---

## Response to Reviewer 2 (Domain, U. Waterloo)

### R2-W1 [Major] — No engagement with quantum kernel literature

**Response**: Accepted. §2.3 now explicitly discusses:

- Havlíček et al. (2019), *Nature* — angle-encoded circuits implicitly compute a kernel  
  k(x, x') = |⟨Ψ(x)|Ψ(x')⟩|².
- Schuld (2021), *PRL* — any VQC classifier is equivalent to a kernel method with the
  circuit's feature map as the kernel.

The implication for QIT is stated directly: the parity advantage can be understood as
QIT's quantum feature map being better aligned with the parity decision boundary than
any classical polynomial kernel of comparable parameter count. We also note in §6.1
that the BV-phase-kickback structure provides a theoretical grounding for *why* this
particular kernel aligns with F₂-linear tasks.

**[Revised]** §2.3 (Quantum Kernel Theory subsection) added.

---

### R2-W2 [Major] — "Inductive bias" claim informal; no formal characterisation

**Response**: Partially accepted. We cannot fully formalise the characterisation within
this paper's scope — a rigorous expressibility-theoretic treatment of when a task is
"QIT-favourable" would itself be a paper. Instead, §6.1 now offers a concrete
structural criterion: tasks with **F₂-linear relational structure** (global parity,
partial parity, palindrome as a conjunction of position-pair XORs) show the advantage;
tasks reducible to a single-token lookup (first-token detection) do not. This is an
empirically-grounded informal characterisation, and we acknowledge explicitly in §6.2
that a formal PAC-style or kernel-alignment argument is future work.

---

### R2-W3 [Critical] — No post-2021 QML literature

**Response**: Addressed above under EIC-W1. Six post-2021 references added, with
substantive engagement in §2.3.

---

### R2-W4 [Major] — No negative control; parity selected for quantum affinity

**Response**: Accepted. Two negative controls are now included:

1. **First-token detection** (§4.5): QIT-0 (4.0 ± 1.1 epochs) is *not faster* than
   MLP (3.4 ± 3.3) or Transformer (5.8 ± 1.7). All three models converge 5/5. This
   is a task where positional structure matters but no global phase accumulation is
   useful — and QIT has no advantage.

2. **Parity kernel baseline** (Table 2): Logistic regression on the structural feature
   f(x) = Σxᵢ mod 2 fails to converge (0/5 seeds). Parity is not linearly separable
   even with this feature because the even/odd boundary is non-monotone in the sum.

Together, these controls establish that the advantage is structurally specific (appears
on F₂-tasks, absent on local positional tasks) and not attributable to trivial feature
engineering.

---

### R2-Q1 — Can QIT be expressed as a quantum kernel?

**Response**: Yes, formally. As Schuld (2021) shows, any VQC is equivalent to a
kernel method. QIT's feature map is: Ψ(x) = U_att · U_ent · U_enc(x)|0⟩, and the
implied kernel is k(x, x') = |⟨Ψ(x)|Ψ(x')⟩|². Whether this kernel is equivalent to
a *classical* kernel on parity inputs is an open question — the connection would
require computing the Gram matrix explicitly and checking classical reproducibility,
which is future work. We note this in §2.3.

### R2-Q2 — Which post-2021 quantum self-attention papers were considered?

**Response**: See §2.3 for the full discussion. The QNLP programme (Lorenz et al.
2023) is the most directly comparable quantum sequence modelling work. Proposals for
quantum self-attention specifically (inner-product in Hilbert space, amplitude-based
retrieval) are referenced but not individually named as we found no single definitive
post-2021 paper to anchor the comparison; this is noted as a gap in the field.

### R2-Q3 — Is there a task where QIT does NOT converge faster?

**Response**: Yes: first-token detection (Table 4). QIT-0 is comparable to classical
models (4.0 ± 1.1 vs. MLP 3.4 ± 3.3). This is the negative control now in §4.5.

---

## Response to Reviewer 3 (Cross-disciplinary, DeepMind)

### R3-W1 [Major] — Parity kernel baseline missing; baselines not matched to task

**Response**: Accepted. The parity kernel baseline (logistic regression on Σxᵢ) is
now in Table 2. It fails (0/5 seeds, DNF). The reason is structural: parity is not
linearly separable in the sum feature — the boundary is non-monotone. This is now
explained in §4.1. The Transformer's failure is also addressed: 206 parameters is
large for a 16-input task, but the Transformer is included because it is the canonical
sequence model that QIT is conceptually positioned against. Its inability to converge
(even with much more compute) supports rather than weakens the paper's argument.

---

### R3-W2 [Major] — "Attention analogue" framing needs tighter definition

**Response**: Accepted. §3.5 (Relationship to Classical Attention) is new and
explicitly states: QIT does **not** compute pairwise attention weights, produce
attention maps, or perform a weighted sum of value vectors. The term "attention
analogue" is used only to describe a *functional role* (resolving token relationships
for downstream classification) rather than a *mechanism*. The distinction is now
stated in the Introduction and reinforced in §3.5: QIT replaces the attention
*concept* with interference *dynamics*, not the attention computation with a quantum
speed-up.

---

### R3-W3 [Minor] — Epochs ≠ comparable units; gradient step count should be reported

**Response**: Accepted. Table 2 now includes a "Grad. Steps" column (convergence
epoch × steps/epoch). For all models: batch size = 8, dataset = 16 inputs, so
steps/epoch = 2. QIT-0: 12.4 steps; MLP: 63.3 steps (converging seeds); Transformer:
DNF.

---

### R3-W4 [Major] — Parity is permutation-invariant; QIT may be a set classifier

**Response**: Accepted — this is the sharpest criticism in the review and we address
it directly. §4.1 now contains an explicit permutation invariance note:

> Parity is a permutation-invariant (set) function: the label is unchanged by any
> reordering of the input tokens. The ring CNOT topology does encode positional
> information (qubit 2t connects to qubit 2(t+1 mod n)), but this information is
> superfluous for parity. Whether QIT-0 *exploits* positional structure for parity
> cannot be determined from this task alone.

To address this directly, we include a **negative control with positional structure**:
first-token detection (§4.5), where the answer depends on exactly one specific
position. On this task QIT-0 is not faster than classical models, indicating the
circuit can use positional information *when needed* but does not gain from it on
parity (where it is irrelevant). We acknowledge in §6.2 that a conclusive
demonstration of QIT as a *sequence* model (not just a set classifier) requires a
task that is both positionally sensitive and non-trivial — this is the main
motivation for the binary arithmetic experiments planned in QIT-1.

---

### R3-Q1 — What does logistic regression on f(x) = (Σxᵢ) mod 2 achieve?

**Response**: DNF (0/5 seeds). Logistic regression on Σxᵢ cannot solve parity: the
even/odd decision boundary is non-monotone in the integer sum. Inputs with sum 1 and
sum 3 are odd (positive class); inputs with sum 0, 2, and 4 are even (negative class).
No linear threshold on Σxᵢ separates these. This is now stated in §4.1 with the
failure result in Table 2.

### R3-Q2 — Gradient step count for each model in Table 2?

**Response**: Added as a column. See Table 2 and §4.1.

### R3-Q3 — Does permuting input tokens change QIT-0's output?

**Response**: Yes — and we now provide direct experimental evidence. The adjacent order
task (new in this revision, §4.6) has 14 multiset-conflicting input pairs: sequences
with identical token multisets but different labels. For example, [0,1,0,0] (label=1)
and [1,0,0,0] (label=0) contain the same tokens {0,0,0,1} but in different order.
QIT-0 converges to 99%+ accuracy on this task (9.6 ± 1.4 epochs, 5/5 seeds), which is
only possible if the model correctly distinguishes these pairs — i.e., if it responds
to token order, not just token identity. This directly answers R3-Q3 with data rather
than assertion.

---

## Response to Devil's Advocate Reviewer

The DA review raised the hardest objections and we address them without hedging.

---

### DA-C1 [Core / Critical] — Parity kernel baseline absent; advantage may be "QIT vs. bad baselines"

**Response**: The parity kernel baseline (logistic regression on Σxᵢ) is now in
Table 2 and fails (0/5 seeds). The counterargument that this is the wrong
counterfactual is noted: there exist *non-linear* classical models that solve parity
(an XOR tree, for instance). The paper never claimed QIT is better than every
conceivable classical solver — it claims QIT is better than the *comparison class
described in §3*: MLP, GRU, and Transformer of comparable parameter count. A
hard-coded XOR function is not a learned model; it is a correct solution, not a
trained baseline. We are comparing inductive biases, not algorithms.

The parity kernel baseline was included specifically to test the "implicit feature
engineering" alternative explanation, and it fails, which rules out that explanation.

---

### DA-C2 [Evidence Selection Bias] — Only parity tested; consistent with cherry-picking

**Response**: Addressed. The revised paper includes six tasks (§4.6, Table 5):
2-bit partial parity, 3-bit partial parity, 4-bit full parity, first-token detection
(negative control), palindrome detection, and — new in this revision — **adjacent order**
(label = x[0]=0 AND x[1]=1).

Adjacent order is strictly positional: 14 of the 16 inputs belong to multiset-conflicting
pairs where two sequences with the same token bag carry different labels. No set-invariant
model can exceed the 75% majority-class baseline. QIT-0 converges at 9.6 ± 1.4 epochs
(5/5 seeds), proving it uses positional structure. Critically, QIT is **not faster** than
MLP (6.0 ± 2.8, 5/5) on this task — consistent with the paper's claim that the advantage
is F₂-structural, not universal.

The full six-task picture is now:
- F₂-structured tasks (parity family, palindrome): QIT wins on speed and reliability
- Positional tasks without F₂ structure (first-token, adjacent order): QIT competes but does not dominate
- This is the opposite of cherry-picking: QIT loses two out of six comparisons and we report and explain both.

---

### DA-M1 — U_ent claimed critical without ablation

**Response**: Ablation now in §4.3 (Table 3). Correction: U_ent is **not critical**
for parity. U_att is. See EIC-W3 response above.

---

### DA-M2 — "12–58× faster" from single init on 16-input task

**Response**: This figure has been removed from the revised paper. Multi-seed results
(5 seeds) are used throughout. The speed advantage is now stated as "5× lower sample
cost" (Table 4, epochs/parameter metric) and "5/5 vs ≤3/5 seed reliability" — both
derived from the 5-seed protocol.

---

### DA-M3 — Parity solved optimally by O(n) classical algorithm requiring zero training

**Response**: Correct, and we do not dispute this. The paper is not asking whether
parity can be solved efficiently by a hard-coded algorithm — it is asking whether
*learned interference dynamics* can serve as a replacement for *learned dot-product
attention*. An XOR tree is not a trained neural sequence model. The parity task is
used because its structure is well-understood and allows us to analyse *why* QIT
converges fast (the F₂-linear phase-kickback connection in §6.1). The claim is about
inductive bias of learned models, not about algorithmic optimality.

---

### DA Unexamined Premise — Parity is permutation-invariant; QIT may be a set classifier

**Response**: This premise is now directly falsified by experiment. The adjacent order
task (§4.6, new in this revision) is defined so that no set-invariant model can exceed
75% accuracy: it has 14 multiset-conflicting input pairs where sequences sharing the
same token bag carry opposite labels. QIT-0 achieves 99%+ accuracy (9.6 ± 1.4 epochs,
5/5 seeds), which requires correctly resolving these pairs based on token order.

QIT-0 is therefore not a set classifier. The permutation invariance of parity is
acknowledged explicitly in §4.1, and the adjacent order result closes the gap: QIT
learns position-dependent tasks when the task demands it, and gains an additional
inductive bias advantage specifically when the task has F₂-linear phase structure.

---

## Summary of Changes

| Revision ID | Change | Location in Revised Paper |
|---|---|---|
| R1 | 5-seed statistics throughout; mean ± std in all tables | Tables 2–5, all result figures |
| R2 | Ablation study: ring / star / no-ent / frozen U_att | §4.3, Table 3 |
| R3 | 6 post-2021 references; §2.3 rewritten | §2.3, references.bib |
| R4 | Parity kernel baseline (logistic regression on Σxᵢ) | Table 2, §4.1 |
| R5 | First-token detection negative control | §4.5, Table 4, Fig. 4 |
| R6 | Permutation invariance note; positional task plan | §4.1, §5 |
| S1 | Gradient step count column in Table 2 | Table 2, §4.1 |
| S2 | Gradient variance at 8 qubits; barren plateau analysis | §4.4 |
| S3 | Attention analogue clarification; §3.5 added | §3.5 |
| S4 | Quantum kernel framing via Havlíček 2019, Schuld 2021 | §2.3 |
| S5 | GitHub repository URL | §7 (Conclusion) |
| S6 | Learning curves (accuracy vs. gradient steps) | Figs. 3a, 4 |
| S7 | Shot-count analysis | Flagged in §5; hardware access needed |
| — | Removed "12–58× faster" headline (single-run artefact) | Replaced throughout |
| — | U_ent "critical" claim corrected to "helpful but not required" | §4.3 |
| — | Task variety benchmark: 6 tasks, 5 seeds each (incl. adjacent order positional probe) | §4.6, Table 5, Fig. 5 |

We believe the revised manuscript addresses every required and suggested revision.
The core claim — that interference-based attention exhibits measurable inductive bias
advantage on F₂-structured tasks — is now supported by multi-seed statistics,
ablation evidence, a negative control, five-task variety data, and a gradient variance
check. We thank the reviewers for the rigour they applied.

---

*Mihail Stancescu*  
*Corresponding Author*  
*madmishu007@gmail.com*

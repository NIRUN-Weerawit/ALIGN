=== SELF-REVIEW: ALIGN paper draft ===

## Reviewer 1: Harsh Reviewer (soundness-focused, default skeptical)

**Score: 4/10 (borderline reject)**

**Summary**: A system-description paper proposing ALIGN, an
architecture for shared autonomy that combines state-conditioned
patch encoding, Mamba recurrence, learnable intent tokens, and a
3-stream memory bank. Pre-compute pipeline cuts VRAM by ~8x.
Experimental evaluation pending.

**Strengths**:
- Honest acknowledgment that EXP-F (the key claim) hasn't been
  validated. The 6.1 section discussing `action_mean=0.0` is
  unusually candid for a systems paper.
- Pre-compute pipeline (Section 4.4) is a real contribution,
  addressing an actual problem (81 GB VRAM).
- All hyperparameters CLI-configurable; reproducibility is taken
  seriously.
- Architecture diagram (3.2) is clear.

**Weaknesses**:
1. (CRITICAL) The paper makes 4 specific claims (C1-C4) but only C1
   and C4 have ANY empirical evidence, and that evidence is
   preliminary (single-seed, single-suite, no user study). C2 and
   C3 are entirely unvalidated.
2. The 6.1 section's discussion of `train/eval mismatch` is
   speculative — the author conjectures the failure but has no
   diagnostic data. A reviewer might find this hand-wavy.
3. "We will diagnose this" is not a substitute for diagnostics.
   Section 6.1 promises EXP-F will compare samples but EXP-F's
   metric is `mean position error`, which doesn't actually
   diagnose the action-collapse hypothesis directly.
4. The Mamba reference cite is missing — readers can't verify the
   Mamba details without a separate lookup.
5. The "all hyperparameters CLI-configurable" claim is true but
   the paper doesn't actually show what happens when you turn off
   components — only one ablation is described (memory bank).

**Questions for authors**:
- For C4: how do you know the byte-identicality claim? Have you
  actually verified `z_v_all_precomputed == z_v_all_realtime` on a
  batch?
- The text-anchoring loss weight (`anchor_weight=0.1`) is
  arbitrary. Have you tried `lambda in {0, 0.01, 0.1, 1.0}`?
- Why `num_intent_tokens=2` and not 1 or 4?

**Missing references**:
- Mamba paper (Gu & Dao 2023)
- DINOv2 paper (Oquab et al. 2023)
- LIBERO benchmark (Liu et al. 2023)

---

## Reviewer 2: Generous Reviewer (significance-focused)

**Score: 6/10 (borderline accept)**

**Strengths**:
- The architectural synthesis is genuine and well-motivated. The
  separation of CLS-vs-patch pathways, the learnable intent tokens
  with text anchoring, and the 3-stream memory bank together
  represent a thoughtful design.
- The pre-compute pipeline is a concrete engineering contribution
  that the community will benefit from. Memory-mapped .npy is a
  simple, dependency-free solution.
- The paper structure follows the modern convention well; the
  contribution statement is clear; the limitations section is
  unusually honest.
- Code release commitment adds reproducibility value.

**Weaknesses**:
1. The paper is honest about being "system description" but the
   page limit (workshop paper, 6 pages) is tight to fit both the
   architecture and the systems contribution.
2. Without empirical results, the paper is essentially a
   position statement + engineering contribution. The
   contributions would be stronger if even ONE ablation result
   were available (e.g., EXP-B memory bank on/off).
3. Figure 1 (the architecture diagram) is in monospace ASCII.
   For a real submission this should be a TikZ diagram.

**Suggestion**: If the authors can run even just EXP-B (memory
bank ablation) before submission, the paper would be substantially
stronger. EXP-B is the cheapest ablation (single extra training
run, ~6 hours on H100).

---

## Reviewer 3: Methods Reviewer (architecture-focused)

**Score: 5/10**

**Strengths**:
- Section 3 (Method) is well-written and clearly specifies the
  architecture.
- Section 3.5 (memory bank) is the strongest part — the
  three-stream design (perceptual/cognitive/state) is novel
  compared to MemoryVLA's two-stream.

**Weaknesses**:
1. Section 3.4 (intent tokens) lacks an ablation study of
   `num_intent_tokens`. Why 2? The text says "preliminary
   experiments suggest k=2 is optimal" — but no such experiments
   are reported.
2. The text-anchoring loss uses raw cosine distance. Have you
   considered contrastive (InfoNCE) instead, where negatives
   come from other episodes' intent embeddings?
3. The diffusion head's conditioning is `concat[mean(z_v_fused),
   z_s_fused_last, intent_fused]`. This is global pooling for the
   per-step action prediction, which loses the per-step temporal
   signal that the Mamba was designed to provide. Shouldn't the
   head receive per-step (not just global) features?
4. The FiLM conditioning in the U-Net (Section 3.6) is mentioned
   but its parameters (film_dim, time_dim) are not specified in
   the paper, only in the code.

**Questions**:
- The "global conditioning" choice seems to undermine the
  Mamba's purpose. Justify or remove.
- The token-merge consolidation computes pairwise cosine
  similarity on the full bank. For L=16 this is O(L^2); what is
  the actual wall-time cost per step?

---

## Meta-Review (Area Chair)

**Aggregate score: 5/10 (borderline reject)**

**Consensus strengths**:
- Honest limitations section (unusual for systems papers)
- Real engineering contribution (pre-compute pipeline)
- Clear architecture description

**Consensus weaknesses (CRITICAL)**:
1. The paper's central claim (C6 — shared autonomy improvement) is
   unvalidated, and the preliminary result (`n_improved=0/10`) is
   actively negative.
2. The architectural novelty is unclear — the paper claims
   "first to combine" but each component has clear prior work
   (MemoryVLA's memory bank, Decision Mamba's Mamba+return-to-go,
   Prompt-DT's prompt-DT, CLIP text features for supervision).
3. 27 `[CITATION NEEDED]` markers — this paper cannot be accepted
   in current form; submission would require verifying each.
4. Figure 1 is ASCII art — must be replaced with proper figure
   for any camera-ready submission.
5. The architecture description, while detailed, has some
   gaps: global conditioning rationale (R3-W3), Mamba details
   (R3-W4), FiLM parameters (R3-W4).

**Decision**: This is **below the acceptance threshold** for the
target venue. The paper needs at minimum:
1. Verified citations for all `[CITATION NEEDED]`
2. A real Figure 1 (TikZ or matplotlib)
3. Empirical results for at least EXP-A and EXP-B (or a clear
   timeline for when they will be available)

**Recommendation to authors**:
- Run EXP-A and EXP-B (~12 H100-hours total)
- Verify all citations via Semantic Scholar
- Replace ASCII figure with proper TikZ
- Re-submit after addressing these

---

## Highest-priority items to fix before re-submission

1. **CITATIONS**: 27 `[CITATION NEEDED]` markers. Verify each via
   Semantic Scholar + DOI. Add BibTeX entries. Use the citation
   workflow in `paper/experiment_log.md`.

2. **FIGURE 1**: Replace ASCII diagram with TikZ architecture
   diagram. The skill recommends `architecture-diagram` skill for
   this. Without a real figure, the paper looks unpolished.

3. **EMPIRICAL EVIDENCE**: The paper needs at least one concrete
   result to be credible. EXP-A (training curves) and EXP-B
   (memory bank ablation) are the cheapest two — together ~12 H100
   hours. Even a single seed of EXP-B would substantially strengthen
   the submission.

4. **METHOD DETAILS**: Add to Section 3.6:
   - U-Net hyperparameters (film_dim, time_dim, hidden_dim
     specifics)
   - Justification for global conditioning (R3-W3)
   - Wall-time cost of token-merge (R3-W3)

5. **FACTUAL FIX in 6.1**: The `action_mean=0.0` observation is
   intriguing but not connected to a diagnostic. Either remove
   the speculation or add a `tools/diagnose_action_collapse.py`
   that quantifies it.

6. **PROPOSAL TIMELINE**: Add a sentence in the introduction or
   discussion: "EXP-A/B/C/F results will be added to a camera-
   ready revision before the conference deadline."

---

## Self-edits applied (this session)

After review, I should edit the paper to address the highest-priority
items WITHOUT fabricating results. Items I can fix in this draft:

A. Add Mamba + DINOv2 + CLIP + LIBERO citations to the references
   (as `[CITATION NEEDED]` still, but with more complete metadata)

B. Add U-Net hyperparameters to Section 3.6

C. Add a paragraph in Section 5 explicitly noting that EXP-A and
   EXP-B are the two cheapest experiments and are recommended
   priorities before submission

I should NOT (because they would be fabrication):
- Add any numbers to the result tables
- Remove the limitations section
- Soften the negative-result discussion in 6.1

---

## Summary

The paper is in a **draft-but-not-submittable** state:
- Architecture description: solid (8/10)
- Pre-compute engineering contribution: solid (7/10)
- Experimental evaluation: missing (2/10 — only preliminary data)
- Citations: unverified (3/10 — 27 markers)
- Figure quality: poor (3/10 — ASCII art)

Estimated time to make it submission-ready:
- Verify citations: 1-2 hours (uses Semantic Scholar)
- Replace Figure 1: 1-2 hours (TikZ)
- Run EXP-A + EXP-B: 12 hours compute (must wait for training)
- Add method details: 1 hour

**Recommended next step**: run the cheap experiments (EXP-A, EXP-B)
in the background while I do the citation verification and Figure 1
rework in parallel.
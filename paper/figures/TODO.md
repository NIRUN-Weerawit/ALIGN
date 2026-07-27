# Figure TODO

The current paper uses an ASCII diagram for Figure 1 (Section 3.2).
This must be replaced before submission.

## Figure 1: Architecture diagram (Section 3.2)

**Current state**: ASCII text diagram, 14 lines, mono font.
**Required state**: TikZ diagram with:
- Color-coded boxes for each component (encoder, patch encoder, Mamba, head)
- Data flow arrows with labeled tensor shapes
- A small inset showing the memory bank's three streams

**Suggested tools**:
- Use the `architecture-diagram` skill (already in Hermes)
- Use `tikz` with `positioning`, `arrows.meta`, `shapes.geometric` libraries
- Compile with `pdflatex`

**Reference**: see `references/writing-guide.md` in research-paper-writing
skill for TikZ templates.

## Figure 2: Training curves (Section 5.1)

**Required state**: line plot of train/loss and val/loss vs epoch, with
3-seed error bars.

**Data source**: `paper/results/exp_a_curves.json` (TBD after EXP-A runs)

## Figure 3: Memory bank ablation (Section 5.2)

**Required state**: bar chart with two bars per metric:
"Without bank" vs "With bank", over `val/pos_mse`, `val/rot_mse`,
`val/grip_acc`.

**Data source**: `paper/results/exp_b_ablation.json` (TBD after EXP-B runs)

## Figure 4: Intent token probing (Section 5.3)

**Required state**: bar chart with 4 bars (z_future, z_past, E_text, random),
showing mean cosine similarity.

**Data source**: `paper/results/exp_c_probing.json` (TBD after EXP-C runs)

## Figure 5: VRAM comparison (Section 5.4)

**Required state**: bar chart comparing peak VRAM (raw frames vs precomputed)
and per-batch wall time. Two-panel: VRAM, wall time.

**Data source**: `paper/results/exp_d_precompute.json` (TBD after EXP-D runs)

## Figure 6: Success rate by switch-at (Section 5.6)

**Required state**: 3 bars (switch-at 0.0, 0.5, 1.0) showing success rate,
with error bars across N episode subsets.

**Data source**: `paper/results/exp_f_shared_autonomy.json` (TBD after EXP-F runs)
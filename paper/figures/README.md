# ALIGN Paper — Figure Sketches (Preview)

This folder contains LaTeX sources for two figures intended for the
ALIGN paper:

| File | Purpose | Section | Type |
|------|---------|---------|------|
| `figure_sketches.tex` | Self-contained LaTeX with all 3 figures | (test file) | TikZ + forest |
| `figure_venn.tex` | Figure 1: Venn diagram of 3 lineages | §1 Introduction | TikZ |
| `figure_treemap.tex` | Figure 2: Tree map of related work | §2 Related Work | forest |
| `figure_linear.tex` | Figure 1 (variant): Linear taxonomy | §1 Introduction | TikZ |

## Compilation

```bash
# Standard
pdflatex figure_sketches.tex

# Or build each figure independently
pdflatex figure_venn.tex
pdflatex figure_treemap.tex
pdflatex figure_linear.tex
```

Required TeX Live packages (all standard):
- `tikz`, `forest`, `xcolor`, `microtype`, `booktabs`, `hyperref`

On Debian/Ubuntu: `apt install texlive-pictures texlive-latex-extra`
On macOS (MacTeX): all included by default.

## Color scheme

All figures use the **Okabe-Ito colorblind-safe palette** so they're
readable in grayscale and accessible to colorblind reviewers:

- Autonomous Manipulation: **blue** (`#0072B2`)
- Shared Autonomy: **orange** (`#E69F00`)
- Task-Semantic Extraction: **green** (`#009E73`)
- ALIGN: **yellow** (`#F0E442`) with red border for emphasis

## Conceptual design

### Figure 1 — Venn diagram (3-circle)

```
            Task-Semantic
            Extraction
                 ▲
                ╱ ╲
               ╱   ╲
              ╱  ★  ╲
             ╱ ALIGN ╲
            ╱ (this   ╲
           ╱  work)    ╲
          ╱             ╲
   Auto ─╲─────────────╱─ Shared
   Manip.   ╲         ╱   Autonomy
            ╲       ╱
             ╲     ╱
              ╲   ╱
               ╲ ╱
                ▼
              (gap)
```

ALIGN sits at the **3-way overlap**. Prior work sits in each lineage's
exclusive region. Pairwise overlaps are **empty** (no prior work spans
even two lineages).

### Figure 2 — Tree map (horizontal)

```
Robotics
├── Autonomous Manipulation (blue)
│   ├── Imitation learning
│   │   ├── BC, ACT, Diffusion Policy (Chi 2023), RT-2, RT-X
│   ├── Foundation VLAs
│   │   ├── OpenVLA, Octo, π₀, GR-1
│   └── Memory-augmented
│       ├── MemoryVLA (2026), RAG-policy
│
├── Shared Autonomy (orange)
│   ├── Goal inference
│   │   ├── POMDP (Javdani), Visual goal (Jonnavittula),
│   │   ├── VLM-mediated (Casper, SAFe-Copilot)
│   ├── Arbitration
│   │   ├── α-blending (Wang & Losey), Forward-reverse (Yoneda),
│   │   ├── Bayesian (RT-V3), Flow-match (Assistron)
│   └── Adaptive α
│       ├── SA-DRL (Reddy), Atan (2024), Lima MPC (2024)
│
└── Task-Semantic Extraction (green)
    ├── Fixed query + attention
    │   ├── BERT [CLS] (2018), Perceiver IO (2021),
    │   ├── Slot Attention (2020), Set Transformer (2019)
    ├── Fixed query + RNN/SSM
    │   ├── Decision Mamba (2024), Prompt-DT (2022), LPDT
    └── Multimodal grounding
        ├── CLIP (Radford 2021), ALIGN (Caron 2021),
        ├── Language-as-supervision

+ ───────────────────────────────────── +
| ALIGN (this work)  ←── 3-way overlap   |
+ ───────────────────────────────────── +
```

### Figure 1 (variant) — Linear taxonomy

Three horizontal rows (one per lineage), each with its prior works listed.
ALIGN appears as a vertical band on the right that crosses all three
rows — visually showing that ALIGN is the only work that touches all
three lineages simultaneously.

This is the simplest layout, good for slides/talks where the Venn
diagram might be too dense.

## Recommended placement in the paper

| Figure | Section | Purpose |
|--------|---------|---------|
| Venn (Fig 1) | §1 Introduction, after contribution bullets | One-glance positioning of ALIGN |
| Tree map (Fig 2) | §2.1 Related Work, before the survey | Enumerate prior work by lineage |
| Linear (Fig 3) | Appendix or supplementary | Talk/slide-friendly version |

The **Venn diagram in §1** is the most important — it's what reviewers
will see first when they ask "where does this fit?".

## How to use in the actual paper draft

When you build the camera-ready paper:

1. Copy the `figure_X.tex` body into a figure environment in the paper.
2. Use `\input{figure_X}` to keep the source modular.
3. The colors and sizes are designed to fit in a single column (NeurIPS,
   ICML, ICLR, COLM) or double column (TPLR).
4. Adjust `\R`, font sizes, and circle radius to match your page width.

## Why this design

The Venn diagram's punchline is **"the 3-way intersection is empty until
ALIGN."** It makes the contribution claim falsifiable: a reviewer can
disagree by pointing to a paper that does occupy all three lineages.
(The current literature doesn't.)

The tree map's punchline is **"ALIGN integrates mechanisms from each
lineage."** It supports the same claim with more detail, useful for
reviewers who want to verify the lineage assignments.

Together they form a single coherent argument:
**ALIGN is at the intersection, and the intersection was empty before.**

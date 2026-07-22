#!/usr/bin/env python3
"""Evaluate ALIGN on LIBERO simulation with all fixes applied.

This script pulls together every fix made during development:

PIPELINE FIXES
- Phase 1a: Full trajectory window (not collapsed to single frame)   [69f9e94]
- Phase 1a: 4D → 3D frame handling (delta_timestamps temporal dim)   [75d1e1a]
- Phase 1a: BF16 autocast without redundant .float() cast            [c2fc2d0]
- Phase 1a: Validation every N epochs                                [c2fc2d0]
- Phase 1b: Mixer unfrozen, InfoNCE on mixer outputs                 [phase logic]
- Phase 2:  3D pose flattening for noise injection                   [61f116f]

DATA FIXES
- Pre-decode LIBERO → HDF5 for num_workers > 0 support              [c15aa0a]
- Both cameras saved in single HDF5                                  [3ce7169]

INFRASTRUCTURE FIXES
- lerobot get_safe_version monkey-patch (version tag bug)            [3464c84, 1e27bdf]
- LIBERO metadata cache from HF Hub API (not raw URLs)               [85cc72c]
- Auto run numbering (run_1, run_2, ...)                             [c2fc2d0]
- chunk_size auto-detection from checkpoint config                   [ab1a402, 0f1ebed]
- torch 2.10.0+cu128 / torchvision 0.25.0 / torchcodec 0.10.0       [35d36c1]
- xformers installed separately after torch to avoid dep conflicts   [35d36c1]
- scripts/check_deps.py for easy verification                        [0835355]
- .gitignore for HDF5, eval artifacts                                [28c9d5e, fd7a47c]

INFERENCE FIXES
- encode_mixed (not encode_raw_all) — proper mixer usage             [781ff7c]
- Full gating signal: α = need × consistency                         [8124ba6]
- encoder_checkpoint flag for proper backbone loading                [inference fix]
- detect chunk_size from checkpoint config                           [781ff7c]
- Precomputed z_sext cache for speed                                 [781ff7c]

SIMULATION EVAL (eval/eval_libero.py)
- OffScreenRenderEnv with BDDL file path                             [eb0a63d]
- All 5 LIBERO suites supported                                      [1927409]
- Video recording with α/Δ/step overlay                              [c5a448a]
- Reduced font size (14 instead of 18)                               [1ec85ac]

KNOWN ISSUES
- Decision head α MAE ~0.22-0.37 — needs better noise variance
- num_workers > 0 hangs with LeRobotDataset on some machines
- Assistant head converges in ~10 epochs, 100+ epochs wasteful
"""

if __name__ == "__main__":
    print(__doc__)
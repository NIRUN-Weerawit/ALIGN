#!/usr/bin/env python3
"""Build immutable, provenance-aware whole-demonstration targets."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.task_context_dataset import build_targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cameras',nargs='+',default=['image','wrist_image'])
    parser.add_argument('--fps',type=float,required=True,help='Declared source rate when timestamps are absent')
    parser.add_argument('--seed',type=int,default=20261020)
    parser.add_argument('--terminal-tail',type=int,default=8)
    parser.add_argument('--action-horizon',type=int,default=8)
    parser.add_argument('--procedure-points',type=int,default=8)
    parser.add_argument('--annotations',help='Optional per-episode JSON with verified success/terminal intervals')
    parser.add_argument('--split-mode', choices=['auto','legacy'], default='auto',
        help='Default: fixed Piper 384/44 for LIBERO Goal; legacy restores the original three-way split')
    args=parser.parse_args()
    build_targets(args.data,args.cache,args.output,cameras=args.cameras,fps=args.fps,
        seed=args.seed,terminal_tail=args.terminal_tail,action_horizon=args.action_horizon,
        procedure_points=args.procedure_points,annotations=args.annotations,split_mode=args.split_mode)


if __name__=='__main__':main()

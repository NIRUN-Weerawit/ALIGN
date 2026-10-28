"""Saved flow-matching options shared by training and checkpoint evaluation."""
import argparse
import math


# Checkpoints predating these options used these settings. Never infer current
# training defaults for a historical model at evaluation or resume time.
LEGACY_FLOW_DEFAULTS = dict(flow_inference_steps=10, flow_solver='euler',
    flow_loss_repeats=1, flow_clip_sample=False, flow_sigma_min=0.)


def flow_policy_options(config):
    return {key: config.get(key, value) for key, value in LEGACY_FLOW_DEFAULTS.items()}


def add_flow_policy_arguments(parser):
    parser.add_argument('--flow-inference-steps', type=int, default=10,
        help='ODE integration steps for flow sampling (Heun uses twice as many network calls)')
    parser.add_argument('--flow-solver', choices=('euler', 'heun'), default='euler')
    parser.add_argument('--flow-loss-repeats', type=int, default=4,
        help='Independent time/noise draws per supervised action chunk, matching diffusion training')
    parser.add_argument('--flow-clip-sample', action=argparse.BooleanOptionalAction, default=True,
        help='Clip terminal normalized actions: [-1,1], with gripper [0,1]')
    parser.add_argument('--flow-sigma-min', type=float, default=0.,
        help='Residual Gaussian noise at the data endpoint; zero gives a clean target')


def validate_flow_policy_arguments(args):
    for key in ('flow_inference_steps', 'flow_loss_repeats'):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.flow_sigma_min) or not 0 <= args.flow_sigma_min < 1:
        raise ValueError('--flow-sigma-min must be finite and in [0, 1)')

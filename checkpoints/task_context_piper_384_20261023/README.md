# Context teacher/student refit on the Piper split

384 training and 44 held-out episodes, exactly matching the Piper ablations.

See [protocol, results and verification](../../reports/task_context_piper_384_20261023/summary.md).

Use `observer_frozen.pt` for new action-policy training. Existing action heads require retraining for this new latent space.

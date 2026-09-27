# Piper X-VLA fine-tuning and supervised inference report

**Updated:** 2026-09-27 (Asia/Tokyo). Code behavior and recording statistics were checked against the current workspace.

**Summary:** We adapted the `xvla-base` checkpoint to one Piper task—**“grab an object and put in a cup”**—using an immutable, explicit subset of pendant-replay demonstrations. Offline evaluation is usable as a learning diagnostic, but it is not evidence of autonomous hardware safety. The current hardware path is operator-gated continuous control: a background process captures observations and predicts actions, while **Hold to send controls** renews a short authorization lease. The runner sends fresh, guard-approved pose and gripper targets only during an authorized hold, with a nominal 20 Hz schedule. Recorded inference rates are approximately **7.3–7.9 Hz**, so sustained 20 Hz policy control has not been achieved. Both camera views can be recorded asynchronously and converted into timestamp-paced review videos.

This report describes the currently implemented workflow and the artifacts present in the workspace. It distinguishes measured results from safety controls and from remaining work.

---

## 1. Scope and current status

| Item | Current status | Evidence |
|---|---|---|
| Task | Single task: `grab an object and put in a cup` | `piper_xvla/live_inference.py` task constant and WebUI task flow |
| Training data | 101 retained pendant-replay episodes, 24,352 frames | `data/piper_replay/xvla_training_manifest.json` |
| Base model | Local X-VLA base checkpoint | `piper_xvla/config/piper_xvla_single_task.json` |
| Fine-tuned checkpoint | `outputs/piper_xvla_single_task/best.pt`, selected at step 18,000 | `outputs/piper_xvla_single_task/offline_eval_best.json` |
| Held-out offline evaluation | Completed on 11 episodes / 2,535 frames | `offline_eval_best.json` |
| Live camera/CAN inference | Continuous background process; diagnostic by default, optional lease-gated command capability | `piper_xvla/live_inference.py`, inference JSONL/log |
| Physical control | Hold-gated policy commands sent from the runner; measured poses change in recordings, but task success is unverified | `live_inference.py`, `webui.py`, recorded frames |
| Control frequency | 20 Hz target; approximately 7.3–7.9 Hz measured in saved sessions | Recording timestamps and `inference_ms` |
| Guard configuration | Editable in the X-VLA panel; validated, persisted, and reloaded before action checks | `action_guard.py`, `/api/xvla/guard` |
| Image recording / video | Both camera images, per-frame metadata, bounded writer queue, H.264 conversion | `inference_recorder.py`, `recording_to_video.py` |
| Autonomous task completion | **Not established** | No validated closed-loop object-placement trial in the available artifacts |

The word **safe** below means *bounded and operator-supervised*, not proven safe for unattended operation or near people.

---

## 2. Demonstration data selection and preservation

### 2.1 Immutable source data

The raw legacy source was left untouched:

```text
~/ALIGN/baselines/data/piper_replay/dataset
```

It contains embedded camera images in Parquet and is approximately 21 GB. Some episode metadata refers to deleted data files. Rather than copying or rewriting the dataset, `discover_intact_replay_episodes()` checks that each referenced Parquet file exists and selects only intact episodes.

This avoided two risks:

1. training against stale metadata or missing episode files; and
2. creating a second large image dataset solely to make episode indices contiguous.

### 2.2 Explicit retained-data manifest

The selection is fixed in:

```text
~/ALIGN/baselines/data/piper_replay/xvla_training_manifest.json
```

Recorded facts:

- **101** retained episodes;
- **24,352** total frames;
- two RGB observations: global and wrist, each 480×640;
- 8-D model proprioception;
- 20-D X-VLA action target;
- explicit original episode IDs; and
- gripper calibration limits derived from retained labels:

```text
minimum = -0.042399998754262924 m
maximum =  0.0478999987244606 m
```

The explicit manifest makes future training reproducible and prevents silently changing the population when raw files are edited.

---

## 3. Representation and label conversion

### 3.1 Raw Piper semantics

Each raw Piper state/action is 10-D:

```text
XYZ (3) + rotation-6D (6) + physical gripper stroke (1)
```

Pendant replay produces measured-future labels: for non-terminal frames, the action is the next measured state.

Piper feedback Euler angles are interpreted with SciPy’s **extrinsic `xyz`** convention. Rotation-6D decoding for command generation and WebUI display now uses `as_euler("xyz")` to invert that encoding. Uppercase `"XYZ"` is a different, intrinsic convention and must not be substituted.

### 3.2 X-VLA target contract

The fine-tuning adapter maps this to X-VLA’s 20-D EE6D bimanual action layout:

```text
active Piper arm (10-D) + inactive arm zeros (10-D)
```

The physical gripper stroke is normalized monotonically to `[0, 1]` using the manifest’s retained-data minimum and maximum. The inactive arm is exactly zero in targets; its predicted magnitude is monitored independently at evaluation and live inference.

### 3.3 Model input contract

The installed X-VLA checkpoint expects:

```text
- global RGB image
- wrist RGB image
- one injected black 224×224 empty camera view
- 8-D proprioception = XYZ (3) + quaternion xyzw (4) + normalized gripper (1)
- task-language tokens, maximum length 64
```

The action remains the 20-D EE6D target. The adapter does not feed raw 20-D action representation back as proprioception.

### 3.4 Numeric cache without copying image payloads

Before training, a compact numeric cache is built at:

```text
~/ALIGN/baselines/data/piper_replay/piper_xvla_prepared_cache.pt
```

`prepare_xvla_cache.py` reads only numeric Parquet columns and caches converted 8-D state and 20-D action labels keyed by global frame index. Images remain in the original source Parquet files. Cache loading validates source root, retained episode IDs, frame count, and calibration against the manifest.

Train and validation sources are opened separately and share the prepared numeric cache. This retains episode separation without making a RAM-heavy unified embedded-image dataset.

---

## 4. Fine-tuning procedure

### 4.1 Configuration

Current configuration:

```text
piper_xvla/config/piper_xvla_single_task.json
```

| Setting | Value |
|---|---:|
| Optimizer steps | 20,000 configured |
| Batch size | 4 |
| Validation episodes | 11 |
| Device | CUDA |
| Action mode | `ee6d` |
| Vision encoder | unfrozen |
| Language encoder | unfrozen |
| Policy transformer | trainable |
| Soft prompts | trainable |
| Base LR | 1e-4 |
| Warmup | 1,000 steps |
| Decay horizon | 20,000 steps |
| Validation/checkpoint interval | 1,000 steps |

The training code explicitly selects BF16, `ee6d`, 64 language tokens, `chunk_size=1`, and `n_action_steps=1`.

### 4.2 Why these settings were explicit

- `ee6d` preserves X-VLA’s intended position, rotation-6D, and BCE gripper semantics. Generic action mode would change that loss contract.
- Both visual and language backbones were deliberately unfrozen for the new embodiment/task adaptation.
- The model has a practical sequence-length limit after image tokens; tokenizer length 64 avoids the sequence-length failure caused by the checkpoint JSON’s larger nominal value.
- One-step targets match the available measured-future labels. This is a data-contract decision, not evidence that one-step closed-loop control is sufficient.
- The built-in X-VLA optimizer parameter grouping and warmup/cosine schedule are used instead of a generic optimizer configuration.

### 4.3 Pre-training gate

Before a full run, the pipeline required a real loaded checkpoint to produce a finite loss on a batch containing the full contract:

```text
state [B,8]
action [B,20]
global image [B,3,480,640]
wrist image [B,3,480,640]
empty view [B,3,224,224]
language tokens [B,64]
```

A finite BF16 smoke-loss of `22.388092041015625` was obtained on the local RTX 4060. This was a compatibility gate only; it was not an accuracy or safety result.

### 4.4 Reproducibility outputs

The trainer writes:

```text
outputs/piper_xvla_single_task/
├── best.pt
├── split.json
└── periodic checkpoints
```

`best.pt` is a custom `torch.save` dictionary containing model weights, optimizer state, scheduler state, step, validation loss, split, and serialized X-VLA configuration. It must be reconstructed with the custom evaluator; it is not a Hugging Face checkpoint directory.

The held-out split is episode-disjoint:

- training: 90 episodes, IDs in `split.json`;
- validation: 11 episodes: `113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 124`.

---

## 5. Offline evaluation

The evaluator reconstructs the model from the serialized X-VLA configuration, restores `checkpoint["model"]`, and evaluates the held-out episodes without using the training episodes.

Artifact:

```text
outputs/piper_xvla_single_task/offline_eval_best.json
```

Checkpoint and evaluation population:

```text
checkpoint step: 18,000
validation frames: 2,535
validation episodes: 11
```

### 5.1 Held-out results

| Metric | Measured value |
|---|---:|
| Position RMSE | **0.01005 m** |
| Rotation-6D RMSE | 0.03126 |
| Gripper MAE, normalized | 0.02639 |
| Inactive-arm RMSE | 0.001823 |
| Position loss | 0.02873 |
| Rotation-6D loss | 0.01641 |
| Gripper loss | 0.14282 |
| Total reported loss | 0.18795 |

The total loss must not be read as metres or degrees. The `ee6d` components are differently scaled: position loss is meter-space MSE with X-VLA’s position weighting, rotation is 6D-space loss, and gripper uses BCE.

### 5.2 Offline action diagnostics

The same artifact reports:

| Diagnostic | Value |
|---|---:|
| Non-finite generated actions | 0 |
| Generated gripper values outside `[0,1]` | 0 |
| Degenerate rotation-6D outputs | 0 |
| Predictions outside target XYZ envelope | 149 frames |
| Maximum predicted XYZ step | 0.04403 m |
| Implied maximum XYZ velocity at 20 Hz | 0.88069 m/s |
| Maximum predicted rotation step | 8.6868° |
| Maximum predicted gripper step | 0.3740 normalized |

The envelope and implied velocity are diagnostics from demonstrations/evaluation—not verified hardware limits. In particular, the 149 target-envelope excursions and 44 mm maximum generated step mean the offline model cannot be treated as directly hardware-safe.

---

## 6. Live inference implementation

### 6.1 Persistent inference process and optional control

`piper_xvla/live_inference.py` loads the policy and tokenizer once, opens both cameras and a Piper CAN connection, and runs until its finite frame count is reached or it is stopped. `--frames 0` means continuous inference. Language tokens are computed once and reused.

Each cycle:

1. captures the global and wrist RGB images and copies measured robot state;
2. converts that state into 8-D proprioception and a normalized active-arm guard reference;
3. runs `predict_action_chunk` and uses the first one-step action;
4. reloads the guard configuration if its file has changed, then validates the prediction;
5. checks the held-control lease and, if authorized and guard-approved, sends one Cartesian target and one gripper target;
6. optionally queues the same images and metadata for recording; and
7. writes inference, guard, control, and timing telemetry to JSONL.

Standalone inference remains **diagnostic by default**. Command capability requires both `--enable-held-control` and `--hold-lease <path>`. The WebUI launches the runner with these flags, but an expired lease permits no policy target transmission. The runner now contains arm enable, CAN/MOVE-P mode setup, `EndPoseCtrl`, and `GripperCtrl` calls; the earlier statement that it had no command capability is obsolete.

The CAN connection exists before Hold is pressed because it is already needed for feedback. Pressing Hold creates a control session; it does **not** start another inference process, thread, model load, or camera connection.

### 6.2 Scheduling and measured throughput

The nominal period is **50 ms / 20 Hz**. Camera reads, inference, guard checks, and command transmission run sequentially in the same loop. Overruns are reported as missed deadlines; old predictions are not repeatedly sent in a catch-up burst. The implementation is a best-effort scheduler, not a real-time control guarantee.

Telemetry includes:

```text
frame, timestamp_monotonic_s, inference_completed_monotonic_s
camera_feedback_acquisition_s, inference_ms, cycle_ms
loop_hz, control_hz, target_hz, deadline_missed
control_active, control_sent, control_release_hold_sent
guard_allowed, alerts, predicted_action20, current_active10
recording (when enabled)
```

The WebUI shows loop/control rates, inference latency, missed deadlines, and fresh prediction status. Saved sessions measured approximately 7.3–7.9 Hz, with mean inference time around 123–134 ms. Policy inference alone therefore exceeds the 50 ms target in these sessions. Sustained 20 Hz policy control remains unverified and would require reducing end-to-end cycle time.

### 6.3 Camera ownership and timestamp limitations

Configured roles remain global `/dev/video14` and wrist `/dev/video8`, both 640×480 at a configured 20 FPS. Device numbering must be checked after reconnecting cameras.

Preview, collection, and inference serialize camera ownership. The standalone collection recorder cannot run concurrently with inference; the new inference recorder reuses already-captured images instead of opening the cameras again.

Guard camera/feedback ages currently use **local acquisition-duration proxies**, not camera-driver or SDK source timestamps. These are not full observation ages at transmission time: inference and setup can add delay after acquisition. Source timestamp validation and a stricter end-to-end freshness check remain work to do.

### 6.4 Editable action guard

`ActionGuard` rejects invalid/nonfinite data, excessive inactive-arm output, workspace violations, gripper values outside `[0,1]`, degenerate rotations, excessive position/rotation/gripper steps, and stale acquisition timing. It never clips a rejected action. Near-limit checks produce warnings and do not themselves reject the action.

The X-VLA panel’s expandable **Guard limits** editor exposes all eight configuration keys. Position-step and acquisition-age inputs are displayed in millimetres and milliseconds; workspace bounds are in metres, rotation is in degrees, and gripper steps are normalized.

**Apply limits** is available during inference once the form is loaded and valid. Applying:

- ends the current hold session and invalidates its late heartbeats;
- validates and atomically persists the complete configuration;
- causes the runner to reload it before a subsequent action check; and
- saves the values for future runs.

A new operator press is required to resume held control. Applying invalid values returns a validation error. The configuration filename retains the historical `camera_only` name; it is also used by the held-control runner.

Current saved configuration at this report update:

| Limit | Value |
|---|---:|
| Workspace minimum XYZ | `[-0.05, -0.01, 0.069]` m |
| Workspace maximum XYZ | `[0.37, 0.33, 0.50]` m |
| Position step, Euclidean norm | 0.15 m |
| Rotation step | 35° |
| Normalized gripper step | 1.0 |
| Inactive-arm output L2 norm | 0.05 |
| Camera acquisition-age proxy | 0.25 s |
| Feedback acquisition-age proxy | 0.25 s |

These are the current editable settings, **not validated physical safety limits**. In particular, a normalized gripper-step limit of 1.0 permits the entire calibrated stroke, and a workspace box does not establish that an orientation/position combination is reachable.

Alerts now include the actual value and threshold, for example:

```text
INACTIVE_ARM_NONZERO: inactive arm L2=0.01050 > max=0.01
GRIPPER_STEP_LIMIT: normalized gripper step=0.94444 > max=0.9;
                   predicted=0.02234; measured=0.96678
WORKSPACE_VIOLATION: X=<value> m outside [<min>, <max>] m
```

Those examples came from earlier settings; they do not describe the current larger limits. Workspace alerts identify the violated axes, warnings include their warning threshold and hard limit, and detailed messages appear in the panel and recent-output log.

### 6.5 Recorded session evidence

Three completed image recordings are present under:

```text
outputs/piper_xvla_single_task/inference_recordings/
```

The following figures were recomputed from each session’s `frames.jsonl` and `recording.json`. Duration includes an estimated final-frame interval; sample rate is `(N−1)/(last timestamp−first timestamp)`. These are rates of saved inference observations, not claims that every cycle sent a command.

| Session | Saved pairs | Duration | Sample rate | Guard allowed | Policy send cycles | Dropped pairs |
|---|---:|---:|---:|---:|---:|---:|
| `20260927_195553_630d4c84` | 1,897 | 239.60 s | 7.92 Hz | 1,896 | 1,221 | 0 |
| `20260927_203114_d159fe31` | 1,585 | 216.84 s | 7.31 Hz | 1,584 | 1,503 | 0 |
| `20260927_203528_54ab5ea9` | 3,148 | 396.89 s | 7.93 Hz | 3,147 | 2,836 | 0 |

| Session | Mean inference | Mean predicted-to-measured XYZ distance | Maximum XYZ distance | Measured XYZ excursion |
|---|---:|---:|---:|---|
| `195553` | 123.33 ms | 0.01066 m | 0.14753 m | `[0.24964, 0.18928, 0.27238]` m |
| `203114` | 133.65 ms | 0.01125 m | 0.09171 m | `[0.25244, 0.18246, 0.20868]` m |
| `203528` | 122.90 ms | 0.00788 m | 0.06620 m | `[0.24175, 0.16253, 0.19784]` m |

Each session has one initial rejected frame with both stale-camera and stale-feedback alerts. The first also has 42 position-step and 39 rotation-step warnings; the second has 25 rotation-step warnings; the third has neither of those step warnings. Release-hold send counts are respectively 5, 1, and 4. All three recorder summaries report no writer error and an empty final queue.

`control_sent=true` records a transmission attempt for a paired pose/gripper update; it is not a controller acknowledgment. The measured excursions show that robot pose changed during the sessions, but do not independently attribute every movement to the model or establish object-placement success. Live predicted-to-current distances also differ from the offline prediction-to-label RMSE in Section 5 and should not be equated.

**Guard provenance limitation:** limits can be edited between runs or during a run. Complete configuration snapshots/checksums are not stored per frame or per recording; reloads are printed to the shared WebUI log, which is replaced at the next run. The historical guard decisions and alert messages are retained, but exact historical limits cannot always be reconstructed. The current configuration table is not a substitute for a pinned per-session guard history.

---

## 7. WebUI control and hardware findings

### 7.1 Hold-to-send authorization

The active flow is **Run diagnostic → Manual Control ON → Hold to send controls**. `Frames = 0` keeps inference running continuously.

1. The operator types `ARM` to enable the Manual Control latch.
2. A press creates a unique held-control session ID.
3. The browser renews the lease approximately every 100 ms; each renewal expires after 350 ms.
4. The server checks feedback, arming, inference state, session identity, prediction freshness/guard status, and concurrent manual motion before writing the lease.
5. The inference process checks the lease before sending guard-approved actions at its achieved cycle rate.

Release, lost focus, a hidden page, a guard rejection, or lease expiry ends held control. A rejected sample latches that session off, so a later accepted prediction cannot silently resume it. Revoked session IDs prevent an in-flight heartbeat from reactivating a released press.

When an initialized runner observes lease loss, it sends one target at the just-measured pose/gripper state to replace the last future target, then keeps inferring. This is a best-effort hold command, **not proof of an immediate physical stop**. Response can be delayed by camera capture, inference, or other ongoing work; a crashed or forcibly killed process cannot perform the final hold. The physical E-stop remains separate.

Competing WebUI motion paths are rejected while the held-control lease is active. Motor/CAN setup is cancellable; the first action after setup comes from a newly acquired observation rather than a prediction made before the setup wait.

### 7.2 Fixed lease and orientation bugs

Two observed software faults were corrected:

- **Heartbeat rejected its own control owner.** Renewal formerly called the same ownership guard as a competing manual command. It now retains feedback/arming checks while allowing renewal of its current X-VLA session. Lease loss during motor setup cancels control without raising the old `held-control lease expired while enabling Piper` exception and terminating inference.
- **Euler convention mismatch.** Feedback used `from_euler("xyz")`, but the command/display conversion used `as_euler("XYZ")`. Both command and X-VLA display now use extrinsic `"xyz"`. A numerical matrix round-trip check matched measured SDK orientation to floating-point precision.

During earlier no-motion diagnosis, the runner had logged 801 policy send cycles while the controller reported `TARGET_LIMIT`, all six drivers enabled, and CAN control active. This demonstrated that button authorization reached command transmission; it did not demonstrate controller acceptance. The orientation mismatch was a concrete conversion defect, but an allowed target can still be unreachable after that fix. Guard acceptance does not perform robot inverse kinematics or collision checking.

### 7.3 Units and boundary distinction

The held-control runner denormalizes gripper output using the training calibration:

```text
0.0 -> -42.4 mm
1.0 -> +47.9 mm
```

It converts XYZ metres to Piper integer units of 1 µm and Euler degrees to units of 0.001°. Controller speed is restricted to **1–10%** and follows the manual speed field included with each heartbeat. Controller speed percentage is not the policy-update frequency.

The current held-control path commands directly from `live_inference.py`; it does **not** issue per-frame HTTP requests to `/api/manual/endpose` and `/api/manual/gripper`.

The manual end-pose endpoint still supports an optional per-axis `max_axis_step_m` bound. However, the hold loop does not inherit the manual jog-step setting or that endpoint’s optional bound: it uses `ActionGuard.max_position_step_m`, a **Euclidean XYZ-distance** limit. The earlier report’s claim that the jog-step field enforced the held-control model boundary is obsolete.

### 7.4 Status panel and separate manual controls

The redesigned compact WebUI includes a robot-status panel that floats when scrolled past. It displays controller/arm/teach/motion/error/driver state, joint feedback, measured EEF XYZ in metres, measured Euler RX/RY/RZ in degrees, and gripper opening in millimetres. EEF and gripper values are cleared when feedback is unavailable. Recent output includes inference stdout/stderr, streamed from the worker’s log rather than leaving undrained subprocess pipes.

Separate manual Cartesian, joint, gripper, marked-pose, replay, enable/disable/mode/reset, and E-stop controls remain available. The optional continuous joint/gripper slider mode uses bounded manual requests and is distinct from the model’s held-control process.

---

## 8. Inference image recording and video export

### 8.1 Recording design

The X-VLA panel’s **Record images** checkbox is checked by default for new UI runs. It is chosen at run startup; changing recording state requires a new run. Standalone recording is opt-in with `--record-images-dir <directory>`.

`InferenceImageRecorder` receives the exact global and wrist RGB arrays already used for the policy observation. It does not reopen cameras or capture an additional frame. A background thread encodes full-resolution JPEGs at quality 95 and writes them to disk.

The queue holds at most **8 frame pairs**. Submission is nonblocking: if the writer falls behind, the pair is dropped and the count is displayed. A writer error disables further image writes and reports the error without turning it into a policy exception. Accepted pairs drain on orderly Stop/SIGTERM/SIGINT, with a final summary written at shutdown. Forced termination or a stalled driver can prevent a complete drain.

JPEG encoding and disk I/O are off the inference loop, but CPU/disk contention can still affect throughput. Inference JSONL writes/flushes remain synchronous. No controlled recording-on versus recording-off overhead benchmark has been performed; zero drops does not prove zero overhead.

### 8.2 Output layout and alignment

Each run gets a unique timestamp/UUID directory:

```text
inference_recordings/<session>/
├── global/00000001.jpg
├── wrist/00000001.jpg
├── frames.jsonl
├── recording.json
└── both.mp4                 # only after video conversion
```

`frames.jsonl` maps each saved pair to its inference frame number and monotonic observation timestamp, with prediction, normalized current state, raw physical `state20`, guard alerts, command flags, and timing data. Filenames can have gaps when recording frames are dropped. `recording.json` holds saved/dropped/queued counts and writer error status. Sessions are preserved; subsequent runs do not overwrite their image folders.

### 8.3 Video conversion script and existing result

`piper_xvla/recording_to_video.py` converts one session or all sessions under the parent directory using ffmpeg/libx264. Default layout is global left, wrist right; `--view global` or `--view wrist` exports a single view. Recorded timestamp differences determine frame durations, preserving gaps and variable inference speed; the last frame’s duration is estimated from the median interval. Output defaults to 30 FPS with repeated frames, not 30 Hz recorded observations.

Existing conversion:

```text
20260927_195553_630d4c84/both.mp4
H.264 / MP4, 1280×480, 30 FPS, 239.60 s, 42,318,529 bytes
```

All three sessions now have a side-by-side `both.mp4`, verified with ffprobe as H.264, 1280×480, 30 FPS. Global view is on the left and wrist view is on the right.

| Session | Saved image pairs used | Video duration | MP4 size |
|---|---:|---:|---:|
| `20260927_195553_630d4c84` | 1,897 | 239.600 s | 42,318,529 bytes |
| `20260927_203114_d159fe31` | 1,585 | 216.867 s | 41,656,307 bytes |
| `20260927_203528_54ab5ea9` | 3,148 | 396.867 s | 76,720,771 bytes |

All recorded image pairs were available. Small differences between timestamp-based durations and encoded durations arise from output-frame timing quantization.

---

## 9. Verification and interpretation limits

Latest implementation checks included Python compilation, JavaScript syntax checking with Node, `git diff --check`, numerical Euler round-trip verification, and actual video conversion followed by ffprobe inspection. These checks do not establish real-time performance or robot task success.

The previous report cited **26 WebUI tests passed**. That is historical verification of an earlier change set, not a fresh test result for the later lease, live guard reload, orientation, recorder, and video changes. The test suite was not rerun as part of this report update.

Established evidence includes reproducible data selection, finite checkpoint-compatible training, episode-disjoint offline evaluation, continuous live inference, logged operator-authorized command attempts, changing measured robot poses, saved image pairs with metadata, and a browser-compatible review video.

Not established: reliable grasp-and-cup task completion, sustained 20 Hz policy control, independently validated reachability/collision protection, hard real-time deadman stopping, or safety for unattended use. Non-throwing SDK calls, permissive guard settings, low offline loss, and high guard-allowed counts do not establish these properties.

---

## 10. Next evaluation work

1. Pin code/checkpoint/calibration and preserve initial guard configuration plus every edit inside each recording session.
2. Validate the corrected orientation convention against measured SDK feedback and controller acceptance across several reachable poses.
3. Compare recording-enabled and recording-disabled cycle timings; optimize camera/inference scheduling before claiming 20 Hz policy updates.
4. Add true camera/feedback source ages and evaluate observation age at transmission, including post-setup reacquisition.
5. Measure release/lease-loss response and controller behavior under normal operation and interrupted camera/inference/writer conditions.
6. Use static-pose probes and pendant-replay sweeps to separate state-conditioned tracking from task-mean output collapse.
7. Validate physical workspace, pose reachability, and step bounds with feedback, then assess low-speed supervised object trials with an independent E-stop.
8. Record task outcomes and controller faults explicitly; pose excursions and command-send counts alone cannot establish object-placement success.

---

## 11. Primary files and operation

All commands below run from `~/ALIGN/baselines`.

```bash
# Start the live WebUI; open http://localhost:8050.
PYTHONPATH=. ~/miniconda3/envs/lerobot/bin/python -m piper_xvla.webui --can can0 --port 8050

# Convert all inference image recordings; existing videos are skipped.
python3 -m piper_xvla.recording_to_video

# Convert one session with a single camera.
python3 -m piper_xvla.recording_to_video \
  outputs/piper_xvla_single_task/inference_recordings/20260927_195553_630d4c84 \
  --view wrist
```

In the UI, use `Frames = 0`, choose **Record images**, and click **Run diagnostic**. Confirm the saved guard limits, arm Manual Control with `ARM`, and hold **Hold to send controls** for supervised transmission. Release ends control while inference/recording continue; **Stop** ends the inference run and drains recording. Backend/runner code updates require restarting the WebUI and starting a new inference process; refresh the page for HTML changes.

```text
Data selection and conversion
  piper_xvla/xvla_dataset.py
  piper_xvla/prepare_xvla_cache.py
  data/piper_replay/xvla_training_manifest.json

Training and evaluation
  piper_xvla/train_xvla_piper.py
  piper_xvla/evaluate_xvla_piper.py
  piper_xvla/config/piper_xvla_single_task.json
  outputs/piper_xvla_single_task/best.pt
  outputs/piper_xvla_single_task/split.json
  outputs/piper_xvla_single_task/offline_eval_best.json

Live inference, control, and safeguards
  piper_xvla/live_inference.py
  piper_xvla/snapshot_adapter.py
  piper_xvla/action_guard.py
  piper_xvla/endpose_control.py
  piper_xvla/config/piper_action_guard.camera_only.json
  piper_xvla/webui.py
  piper_xvla/webui.html
  outputs/piper_xvla_single_task/webui_camera_only.jsonl
  outputs/piper_xvla_single_task/webui_camera_only.log

Image recording and video conversion
  piper_xvla/inference_recorder.py
  piper_xvla/recording_to_video.py
  outputs/piper_xvla_single_task/inference_recordings/<session>/
```

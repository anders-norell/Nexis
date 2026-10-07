"""Task environment for the Nexis PoC.

An 8x8 visual field split into 16 patches of 2x2 pixels. A target pixel
drifts across the field for T steps. The agent must report the target's
exact pixel at the final step. The periphery only hints at the patch;
the position inside the patch is visible only in the fovea, so the agent
must keep attention on the target to answer.

Two views of the field are produced every step:
  periphery : (16,)  patch means + heavy noise   -> always available, weak
  fovea     : (16,4) patch pixels + light noise  -> only what attention gates in

Captured trials add a distractor flash: for FLASH_LEN steps a patch away
from the target gets a bottom-up salience spike. The model adds this spike
to its attention logits, so attention is pulled there whether the agent
wants it or not.
"""
import numpy as np

FIELD, PATCH = 8, 2
GRID = FIELD // PATCH          # 4 x 4 patches
N_PATCH = GRID * GRID          # 16
T_STEPS = 12
FLASH_LEN = 2
SD_PERIPHERY = 0.20
SD_FOVEA = 0.30
MOVE_PROB = 0.5
# v2: internal attention drift and a coarse attention trace
DRIFT_RHO = 0.8
DRIFT_SD = 2.5          # stationary SD of the drift on each attention logit
TRACE_SD = 0.1          # noise on the 4-quadrant attention trace


def patch_of(r, c):
    return (r // PATCH) * GRID + (c // PATCH)


def sample_batch(B, rng, p_captured=0.5):
    """Return a dict of arrays for B episodes."""
    pos = np.zeros((B, T_STEPS, 2), dtype=int)
    pos[:, 0] = rng.integers(0, FIELD, size=(B, 2))
    moves = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    for t in range(1, T_STEPS):
        step = moves[rng.integers(0, 4, size=B)] * (rng.random((B, 1)) < MOVE_PROB)
        pos[:, t] = np.clip(pos[:, t - 1] + step, 0, FIELD - 1)

    target_patch = (pos[..., 0] // PATCH) * GRID + (pos[..., 1] // PATCH)  # (B,T)

    # Pixel-level image per step, grouped by patch: (B,T,16,4)
    img = np.zeros((B, T_STEPS, N_PATCH, PATCH * PATCH))
    sub = (pos[..., 0] % PATCH) * PATCH + (pos[..., 1] % PATCH)
    b_idx, t_idx = np.meshgrid(np.arange(B), np.arange(T_STEPS), indexing="ij")
    img[b_idx, t_idx, target_patch, sub] = 1.0

    periphery = img.mean(axis=-1) + rng.normal(0, SD_PERIPHERY, size=(B, T_STEPS, N_PATCH))
    fovea = img + rng.normal(0, SD_FOVEA, size=img.shape)

    # Distractor flashes
    captured = rng.random(B) < p_captured
    flash = np.zeros((B, T_STEPS, N_PATCH))
    flash_patch = np.full(B, -1)
    flash_start = np.full(B, -1)
    for b in np.where(captured)[0]:
        t0 = rng.integers(4, T_STEPS - FLASH_LEN)      # never at start or final step
        # pick a patch that is not the target patch during the flash
        bad = set(target_patch[b, t0:t0 + FLASH_LEN + 1])
        choices = [p for p in range(N_PATCH) if p not in bad]
        fp = rng.choice(choices)
        flash[b, t0:t0 + FLASH_LEN, fp] = 1.0
        flash_patch[b], flash_start[b] = fp, t0

    target_pixel = pos[..., 0] * FIELD + pos[..., 1]  # (B,T) 64 classes

    # v2: AR(1) drift on each patch's attention logit (unknown to the controller)
    drift = np.zeros((B, T_STEPS, N_PATCH))
    innov = DRIFT_SD * np.sqrt(1 - DRIFT_RHO ** 2)
    drift[:, 0] = rng.normal(0, DRIFT_SD, size=(B, N_PATCH))
    for t in range(1, T_STEPS):
        drift[:, t] = DRIFT_RHO * drift[:, t - 1] + rng.normal(0, innov, size=(B, N_PATCH))
    trace_noise = rng.normal(0, TRACE_SD, size=(B, T_STEPS, 4))

    return dict(
        periphery=periphery, fovea=fovea, flash=flash,
        target_patch=target_patch, target_pixel=target_pixel, captured=captured,
        flash_patch=flash_patch, flash_start=flash_start,
        drift=drift, trace_noise=trace_noise,
    )

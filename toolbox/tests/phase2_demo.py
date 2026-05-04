"""End-to-end Phase 2 demo.

Pipeline:
  1. Pick a 'true' unit-param vector for the ball-and-stick cell.
  2. Run jaxley forward on it -> v_true (mV).
  3. Z-score v_true and stuff it into a (B, T, 3) tensor — the same shape
     the dataloader hands the CNN with probsSelect=[0,1,2], stimsSelect=[0],
     serialize_stims=True.  Probes 1 and 2 are filled with mild noise so the
     CNN sees a realistic 3-channel input.
  4. Build an *untrained* CNN matching the m8lay_vs3 design (4-output for
     ball_and_stick) and forward the input -> pred_unit.
  5. Build HybridLoss, compute channel + voltage components.
  6. Print pred_unit / pred_phys, sim trace stats vs true trace stats, and
     each loss term.

Untrained CNN means pred_phys is random, so v_sim won't match v_true —
that's the point: it visualises the gradient signal HybridLoss feeds back
to the optimizer on iteration 0.

Run inside the neuroninverter_jaxley conda env:
    python -m toolbox.tests.phase2_demo
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

import jax
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
rng = np.random.default_rng(0)

from toolbox import JaxleyBridge                                          # noqa: E402
from toolbox.HybridLoss import HybridLoss                                 # noqa: E402
from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS     # noqa: E402
import toolbox.jaxley_cells.ball_and_stick as bas                         # noqa: E402
from toolbox.Model import MyModel                                         # noqa: E402


# ------------------------------------------------------------------
# Knobs
# ------------------------------------------------------------------

# Keep the sim short so this fits in a single-machine demo (~10 s).
bas._T_MAX = 50.0       # ms   ->  T = 500 bins at dt=0.1 ms
JaxleyBridge.clear_cache()

B = 2                   # batch size
NUM_PROBES = 3          # probsSelect=[0,1,2] in m8lay_vs3
P = len(PARAM_KEYS)     # 4 for ball_and_stick

# phys_par_range: [center, log10_halfspan, unit].  Identical to the layout
# stored in sum_train.yaml['input_meta']['phys_par_range'].  unit=0 -> center.
PHYS_PAR_RANGE = [[_DEFAULTS[k], 0.5, "S/cm^2"] for k in PARAM_KEYS]


# ------------------------------------------------------------------
# 1) Ground-truth trace from a known unit-param vector
# ------------------------------------------------------------------

print("=" * 72)
print("STEP 1: build ground truth via jaxley")
print("=" * 72)

# unit=0 means phys=defaults — clean HH spiking trace.
true_unit = torch.zeros(B, P, dtype=torch.float64)

# Convert unit -> phys exactly the way HybridLoss does, so we can show the
# physical values that fed jaxley.
centers  = np.asarray([row[0] for row in PHYS_PAR_RANGE])
logspans = np.asarray([row[1] for row in PHYS_PAR_RANGE])
true_phys = centers * np.power(10.0, true_unit.numpy()[0] * logspans)

print(f"  true_unit  (B={B}, P={P}) = {true_unit[0].numpy()}")
print(f"  true_phys  (P={P})       = {true_phys}")
for k, v in zip(PARAM_KEYS, true_phys):
    print(f"      {k:<12s} = {v:.6f} S/cm^2")

# Pure jaxley forward — no autograd, just to grab v_true (mV).
true_phys_t = torch.tensor(centers * np.power(10.0, true_unit.numpy() * logspans))
with torch.no_grad():
    v_true_mV = JaxleyBridge.simulate_batch(true_phys_t, "ball_and_stick")
print(f"  v_true_mV  shape = {tuple(v_true_mV.shape)}, "
      f"min={v_true_mV.min().item():.2f} max={v_true_mV.max().item():.2f}")


# ------------------------------------------------------------------
# 2) Build the dataloader-shaped input  (B, T, NUM_PROBES) z-scored
# ------------------------------------------------------------------

print()
print("=" * 72)
print("STEP 2: pack v_true into the (B, T, 3) z-scored shape the CNN expects")
print("=" * 72)

T = v_true_mV.shape[-1]  # number of time bins jaxley emits
# Per-sample z-score on the soma trace, matching format_bbp3_for_ML.py.
v_soma = v_true_mV[:, 0, :]                                # (B, T)
v_soma_z = (v_soma - v_soma.mean(dim=-1, keepdim=True)) / \
           (v_soma.std(dim=-1, keepdim=True) + 1e-6)        # (B, T)

# Probes 1 and 2 — fill with mild gaussian noise (realistic-ish stand-ins
# for axon/dendrite that the CNN normally sees but HybridLoss ignores).
images = torch.zeros(B, T, NUM_PROBES, dtype=torch.float64)
images[..., 0] = v_soma_z
images[..., 1] = torch.from_numpy(rng.standard_normal((B, T)) * 0.5)
images[..., 2] = torch.from_numpy(rng.standard_normal((B, T)) * 0.5)
print(f"  images shape = {tuple(images.shape)}  (B, T, probes)")
print(f"  images[:,:,0] (soma): mean={images[:,:,0].mean().item():.3e} "
      f"std={images[:,:,0].std().item():.3f}  <- z-scored")


# ------------------------------------------------------------------
# 3) Untrained CNN, ball-and-stick output dim = 4
# ------------------------------------------------------------------

print()
print("=" * 72)
print("STEP 3: untrained CNN forward -> pred_unit")
print("=" * 72)

# m8lay_vs3 design, but smaller FC dims for speed.
model_hpar = {
    "inputShape":           [T, NUM_PROBES],
    "outputSize":           P,            # 4 for ball_and_stick
    "num_cnn_blocks":       2,
    "conv_block": {
        "filter": [16, 32],
        "kernel": [4, 4],
        "pool":   [4, 4],
    },
    "instance_norm_slot": -9,
    "layer_norm":         False,
    "batch_norm_cnn_slot": 3,
    "batch_norm_flat":    True,
    "fc_block": {
        "dims":    [64, 32],
        "dropFrac": 0.0,
    },
}
# MyModel.__init__ has an explicit fp32 test tensor; build under fp32
# default, then cast the whole thing to fp64 to match the rest of the demo.
_prev = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
cnn = MyModel(model_hpar, verb=0)
torch.set_default_dtype(_prev)
cnn = cnn.double()
cnn.eval()
n_params = sum(p.numel() for p in cnn.parameters())
print(f"  CNN params = {n_params:,}, output dim = {P}")

# Trainer reshapes (B, T, C) -> (B, C, T) inside Model.forwardCnnOnly via
# .view, which expects the channel-first layout.  Let's match that contract.
images_chan_first = images.permute(0, 2, 1).contiguous()   # (B, C, T)
pred_unit = cnn(images_chan_first)                          # (B, P)
print(f"  pred_unit shape = {tuple(pred_unit.shape)}")
print(f"  pred_unit[0]    = {pred_unit[0].detach().numpy()}")


# ------------------------------------------------------------------
# 4) HybridLoss
# ------------------------------------------------------------------

print()
print("=" * 72)
print("STEP 4: HybridLoss(pred_unit, true_unit, images)")
print("=" * 72)

loss_fn = HybridLoss(
    cell_name      = "ball_and_stick",
    phys_par_range = PHYS_PAR_RANGE,
    channel_weight = 1.0,
    voltage_weight = 0.1,
)
loss_fn = loss_fn.double()

# We want to show channel and voltage components separately too, so we
# call the internal helpers as well.
pred_unit_grad = pred_unit.detach().clone().requires_grad_(True)
ch_only = HybridLoss("ball_and_stick", PHYS_PAR_RANGE, 1.0, 0.0).double()
v_only  = HybridLoss("ball_and_stick", PHYS_PAR_RANGE, 0.0, 1.0).double()

L_ch    = ch_only(pred_unit_grad, true_unit, images)
L_v     = v_only( pred_unit_grad, true_unit, images)
L_total = loss_fn(pred_unit_grad, true_unit, images)

print(f"  channel MSE          = {L_ch.item():.6e}     (pred_unit vs true_unit, both unit-space)")
print(f"  voltage MSE          = {L_v.item():.6e}     (z(v_sim)_soma vs z(v_true)_soma)")
print(f"  total (1.0*ch+0.1*v) = {L_total.item():.6e}")
print(f"  expected total       = {1.0*L_ch.item() + 0.1*L_v.item():.6e}")


# ------------------------------------------------------------------
# 5) What did jaxley produce on the CNN's prediction?
# ------------------------------------------------------------------

print()
print("=" * 72)
print("STEP 5: pred_unit -> pred_phys -> jaxley -> v_sim")
print("=" * 72)

with torch.no_grad():
    pred_phys = loss_fn._unit_to_phys(pred_unit_grad.detach())
    v_sim_mV = JaxleyBridge.simulate_batch(pred_phys, "ball_and_stick")
print(f"  pred_phys[0]       = {pred_phys[0].numpy()}")
print(f"  v_sim_mV shape     = {tuple(v_sim_mV.shape)}")
print(f"  v_sim_mV[0,0]  min/max = {v_sim_mV[0,0].min().item():.2f} / "
      f"{v_sim_mV[0,0].max().item():.2f} mV")
print(f"  v_true_mV[0,0] min/max = {v_true_mV[0,0].min().item():.2f} / "
      f"{v_true_mV[0,0].max().item():.2f} mV")

# Sample a handful of time bins side-by-side.
idx = np.linspace(0, T - 1, 8, dtype=int)
print(f"\n  trace sample (sample 0, soma) at bins {idx.tolist()}:")
print(f"    v_true_mV : {v_true_mV[0,0,idx].numpy()}")
print(f"    v_sim_mV  : {v_sim_mV[0,0,idx].numpy()}")
print(f"    diff (mV) : {(v_sim_mV[0,0,idx] - v_true_mV[0,0,idx]).numpy()}")


# ------------------------------------------------------------------
# 6) Backward — confirm gradients reach the CNN
# ------------------------------------------------------------------

print()
print("=" * 72)
print("STEP 6: backward — gradient on CNN's first conv weight")
print("=" * 72)

cnn.zero_grad()
# Re-run with grad-tracked CNN params this time.
pred_unit2 = cnn(images_chan_first)
L_total2 = loss_fn(pred_unit2, true_unit, images)
L_total2.backward()
first_conv = next(cnn.cnn_block[0].parameters())
gnorm = first_conv.grad.norm().item() if first_conv.grad is not None else float("nan")
print(f"  loss              = {L_total2.item():.6e}")
print(f"  ||grad(conv0.W)|| = {gnorm:.3e}")
print(f"  finite grads?     = {torch.isfinite(first_conv.grad).all().item()}")
print()
print("DONE.")

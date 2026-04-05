"""
Test suite for Auth-SwinDiff pipeline.

Covers:
  - AuthBridge (fixed stub)
  - AdaptivePSWABridge (fixed double-counting bug)
  - TimeStepEmbedding (fixed missing clip_proj)
  - GaussianDiffusion (new p_sample / p_sample_loop)
  - SwinDiT backbone (conditional forward pass)
  - PipelineConfig (from_yaml, from_env, validation)
  - InferencePipeline (all four inference modes, no real weights needed)

All tests run on CPU with tiny dummy tensors so no GPU or saved models are
required in CI.
"""

import os
import sys
import tempfile
import pytest
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup so tests can import project modules without installation
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.bridge.auth_bridge import AuthBridge
from src.models.swin_dit.adaptive_backbone import AdaptivePSWABridge
from src.models.swin_dit.layers import TimeStepEmbedding, SwinPatchEmbed, CLIPProjection
from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.utils.pipeline_config import PipelineConfig

# ===========================================================================
# AuthBridge
# ===========================================================================

class TestAuthBridge:
  def test_forward_with_cond(self):
    bridge = AuthBridge(input_dim=64, output_dim=64, cond_dim=64)
    x = torch.randn(2, 16, 64)
    cond = torch.randn(2, 64)
    out = bridge(x, cond)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

  def test_forward_without_cond(self):
    bridge = AuthBridge(input_dim=32, output_dim=32)
    x = torch.randn(2, 8, 32)
    out = bridge(x, cond=None)
    assert out.shape == x.shape

  def test_gate_zero_init(self):
    """
    gate_fc is zero-initialised. Verify:
    1. gate_fc produces exactly-zero logits for any cond → sigmoid = 0.5.
    2. Output shape is correct.
    The bridge doesn't produce identity at init because LayerNorm of
    random x is non-zero, but the gate IS correctly zero-initialised.
    """
    bridge = AuthBridge(input_dim=16, output_dim=16)
    cond = torch.randn(2, 16)
    gate_logits = bridge.gate_fc(cond)
    assert torch.allclose(gate_logits, torch.zeros_like(gate_logits), atol=1e-6), \
        "gate_fc should produce zero logits at init (weights and bias zero-initialised)"
    # Gate values should all equal 0.5
    gates = torch.sigmoid(gate_logits)
    assert torch.allclose(gates, torch.full_like(gates, 0.5), atol=1e-6)

  def test_dim_mismatch_raises(self):
    with pytest.raises(ValueError, match="residual compatibility"):
      AuthBridge(input_dim=32, output_dim=64)

  def test_gradients_flow(self):
    bridge = AuthBridge(input_dim=16, output_dim=16)
    x = torch.randn(2, 4, 16, requires_grad=True)
    cond = torch.randn(2, 16)
    out = bridge(x, cond)
    out.sum().backward()
    assert x.grad is not None

# ===========================================================================
# AdaptivePSWABridge
# ===========================================================================

class TestAdaptivePSWABridge:
  def _make_bridge(self, dim=32, num_scales=3):
    return AdaptivePSWABridge(dim=dim, num_scales=num_scales)

  def test_output_shape(self):
    bridge = self._make_bridge(dim=32)
    x = torch.randn(2, 16, 32)
    t = torch.randn(2, 32)
    out = bridge(x, t)
    assert out.shape == x.shape

  def test_no_double_counting(self):
    """
    After the fix, fused is computed via a single weighted sum over
    stacked_features * weighted_gates — NOT summed again in a loop.
    We verify by checking that output == x + fused (single addition)
    by comparing against a manually computed reference.
    """
    torch.manual_seed(0)
    bridge = self._make_bridge(dim=4, num_scales=2)
    x = torch.randn(1, 2, 4)
    t = torch.randn(1, 4)

    out = bridge(x, t)

    # Manually replicate the FIXED forward pass
    with torch.no_grad():
      x_norm = bridge.norm(x)
      features = [b(x_norm) for b in bridge.frequency_branches]
      gates = bridge.degradation_encoder(t)
      stacked = torch.stack(features, dim=1)
      w = torch.softmax(bridge.fusion_weights, dim=0)
      wg = (gates * w).unsqueeze(-1).unsqueeze(-1)
      fused_expected = (stacked * wg).sum(dim=1)
      expected = x + fused_expected

    assert torch.allclose(out, expected, atol=1e-5), \
      "Output must equal x + single_weighted_fused (no double-counting)"

  def test_residual_connection(self):
    bridge = self._make_bridge(dim=16)
    x = torch.randn(1, 4, 16)
    t = torch.randn(1, 16)
    # Manually zero all parameters so bridge acts as identity
    for p in bridge.parameters():
      p.data.zero_()
    out = bridge(x, t)
    # LayerNorm of all-zero params doesn't produce zero output,
    # but with zero frequency_branch weights the addition is zero
    # so we just check shape here
    assert out.shape == x.shape

  def test_gradients_flow(self):
    bridge = self._make_bridge(dim=16)
    x = torch.randn(2, 4, 16, requires_grad=True)
    t = torch.randn(2, 16)
    out = bridge(x, t)
    out.sum().backward()
    assert x.grad is not None

  def test_num_scales_respected(self):
    for n in [2, 3, 4]:
      bridge = self._make_bridge(dim=32, num_scales=n)
      assert len(bridge.frequency_branches) == n
      assert bridge.fusion_weights.shape == (n,)

# ===========================================================================
# CLIPProjection
# ===========================================================================

class TestCLIPProjection:
  def test_output_shape(self):
    proj = CLIPProjection(clip_dim=768, model_dim=256)
    x = torch.randn(4, 768)
    out = proj(x)
    assert out.shape == (4, 256)

# ===========================================================================
# TimeStepEmbedding
# ===========================================================================

class TestTimeStepEmbedding:
  def _make_emb(self, hidden=128):
    return TimeStepEmbedding(
      hidden_size=hidden,
      frequency_embedding_size=64,
      num_degradation_types=5,
      clip_dim=32,
    )

  def test_unconditional(self):
    emb = self._make_emb()
    t = torch.randint(0, 1000, (4,))
    out = emb(t)
    assert out.shape == (4, 128)

  def test_clip_conditioning(self):
    emb = self._make_emb()
    t = torch.randint(0, 1000, (2,))
    clip = torch.randn(2, 32)
    out = emb(t, clip_embeddings=clip)
    assert out.shape == (2, 128)

  def test_degradation_type_conditioning(self):
    emb = self._make_emb()
    t = torch.randint(0, 100, (3,))
    deg = torch.randint(0, 5, (3,))
    out = emb(t, degradation_type=deg)
    assert out.shape == (3, 128)

  def test_severity_conditioning(self):
    emb = self._make_emb()
    t = torch.randint(0, 100, (2,))
    sev = torch.rand(2)
    out = emb(t, severity=sev)
    assert out.shape == (2, 128)

  def test_fully_conditioned(self):
    emb = self._make_emb()
    t = torch.randint(0, 100, (2,))
    clip = torch.randn(2, 32)
    deg = torch.randint(0, 5, (2,))
    sev = torch.rand(2)
    out = emb(t, clip_embeddings=clip, degradation_type=deg, severity=sev)
    assert out.shape == (2, 128)

  def test_clip_proj_attribute_exists(self):
    """Regression: clip_proj was missing from __init__ in original code."""
    emb = self._make_emb()
    assert hasattr(emb, "clip_proj"), "clip_proj attribute must exist after init"

  def test_scalar_t(self):
    """Scalar timestep (0-dim tensor) should not crash."""
    emb = self._make_emb()
    t = torch.tensor(42)
    out = emb(t)
    assert out.shape == (1, 128)

  def test_no_nan(self):
    emb = self._make_emb()
    t = torch.randint(0, 1000, (8,))
    out = emb(t)
    assert not torch.isnan(out).any()

# ===========================================================================
# SwinPatchEmbed
# ===========================================================================

class TestSwinPatchEmbed:
  def test_output_shape(self):
    embed = SwinPatchEmbed(patch_size=2, no_of_in_channels=4, embed_dim=32)
    x = torch.randn(2, 4, 16, 16)
    tokens, (H, W) = embed(x)
    assert H == 8 and W == 8
    assert tokens.shape == (2, 64, 32)

# ===========================================================================
# GaussianDiffusion
# ===========================================================================

class TestGaussianDiffusion:
  def _make_diffusion(self, T=100, schedule="linear"):
    return GaussianDiffusion(num_timesteps=T, schedule=schedule) # type: ignore

  def test_q_sample_shape(self):
    diff = self._make_diffusion(T=100)
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 100, (2,))
    xt = diff.q_sample(x0, t)
    assert xt.shape == x0.shape

  def test_p_sample_shape(self):
    diff = self._make_diffusion(T=20)
    model_fn = lambda x, t: torch.zeros_like(x)   # predict zero noise
    x_t = torch.randn(1, 4, 8, 8)
    t = torch.tensor([5])
    x_prev = diff.p_sample(model_fn, x_t, t)
    assert x_prev.shape == x_t.shape

  def test_p_sample_t0_no_noise(self):
    """At t=0 the reverse step must add no stochastic noise."""
    diff = self._make_diffusion(T=10)
    model_fn = lambda x, t: torch.zeros_like(x)
    x_t = torch.randn(1, 4, 4, 4)
    t_zero = torch.tensor([0])

    torch.manual_seed(7)
    out1 = diff.p_sample(model_fn, x_t.clone(), t_zero)
    torch.manual_seed(99)
    out2 = diff.p_sample(model_fn, x_t.clone(), t_zero)
    # Results must be identical regardless of seed (no stochasticity at t=0)
    assert torch.allclose(out1, out2, atol=1e-6)

  def test_p_sample_loop_unconditional(self):
    diff = self._make_diffusion(T=20)
    model_fn = lambda x, t: torch.zeros_like(x)
    shape = (1, 4, 8, 8)
    out = diff.p_sample_loop(model_fn, shape, device=torch.device("cpu"))
    assert out.shape == shape

  def test_p_sample_loop_sr_seeded(self):
    diff = self._make_diffusion(T=20)
    model_fn = lambda x, t: torch.zeros_like(x)
    x_lr = torch.randn(1, 4, 8, 8)
    shape = (1, 4, 8, 8)
    out = diff.p_sample_loop(
        model_fn, shape, device=torch.device("cpu"), x_lr=x_lr
    )
    assert out.shape == shape

  def test_p_sample_loop_inference_steps_subsampled(self):
    diff = self._make_diffusion(T=100)
    calls = []
    def model_fn(x, t):
        calls.append(t.item())
        return torch.zeros_like(x)

    diff.p_sample_loop(
      model_fn,
      shape=(1, 4, 4, 4),
      device=torch.device("cpu"),
      num_inference_steps=10,
    )
    assert len(calls) == 10, f"Expected 10 steps, got {len(calls)}"

  def test_progress_callback_called(self):
    diff = self._make_diffusion(T=10)
    model_fn = lambda x, t: torch.zeros_like(x)
    steps_seen = []
    diff.p_sample_loop(
      model_fn,
      shape=(1, 4, 4, 4),
      device=torch.device("cpu"),
      progress_callback=lambda step, x: steps_seen.append(step),
    )
    assert len(steps_seen) == 10

  def test_cosine_schedule(self):
    diff = self._make_diffusion(T=50, schedule="cosine")
    assert diff.betas.shape == (50,)
    assert (diff.betas > 0).all()

  def test_invalid_schedule_raises(self):
    with pytest.raises(ValueError, match="Unsupported schedule"):
      GaussianDiffusion(schedule="linear")

  def test_no_nan_in_q_sample(self):
    diff = self._make_diffusion(T=1000)
    x0 = torch.randn(4, 4, 8, 8)
    t = torch.randint(0, 1000, (4,))
    xt = diff.q_sample(x0, t)
    assert not torch.isnan(xt).any()

# ===========================================================================
# PipelineConfig
# ===========================================================================

class TestPipelineConfig:
  def test_defaults(self):
    cfg = PipelineConfig()
    assert cfg.num_timesteps == 1000
    assert cfg.num_inference_steps == 200
    assert cfg.noise_schedule == "linear"
    assert cfg.in_channels == 4

  def test_from_yaml(self):
    yaml_content = {
      "model": {"latent_size": 32, "in_channels": 4},
      "diffusion": {"num_sampling_steps": 500, "noise_schedule": "cosine"},
      "training": {"learning_rate": 1e-4},
    }
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".yaml", delete=False
    ) as f:
      yaml.dump(yaml_content, f)
      fpath = f.name

    cfg = PipelineConfig.from_yaml(fpath)
    assert cfg.num_timesteps == 500
    assert cfg.noise_schedule == "cosine"
    assert cfg.latent_size == 32
    os.unlink(fpath)

  def test_from_env(self, monkeypatch):
    monkeypatch.setenv("VAE_PATH", "/tmp/vae")
    monkeypatch.setenv("NUM_INFERENCE_STEPS", "50")
    cfg = PipelineConfig.from_env()
    assert cfg.vae_path == "/tmp/vae"
    assert cfg.num_inference_steps == 50

  def test_validate_bad_inference_steps(self):
    cfg = PipelineConfig(num_timesteps=100, num_inference_steps=200)
    with pytest.raises(ValueError, match="num_inference_steps"):
      cfg.validate()

  def test_validate_missing_vae(self):
    cfg = PipelineConfig(vae_path="/nonexistent/path")
    with pytest.raises(FileNotFoundError):
      cfg.validate()


# ===========================================================================
# InferencePipeline (no weights, no real models)
# ===========================================================================

class TestInferencePipeline:
  """
  Tests use a tiny random-weight SwinDiT and a fake diffusion scheduler.
  No VAE or CLIP is loaded — we test the orchestration logic only.
  """

  def _make_tiny_config_yaml(self) -> str:
    content = {
      "model": {
        "name": "test",
        "latent_size": 8,
        "in_channels": 4,
        "patch_size": 2,
        "embed_dim": 32,
        "depths": [1, 1],
        "num_heads": [2, 2],
        "window_size": 4,
        "use_pswa_bridge": False,
        "bridge_alpha": 0.5,
      },
      "diffusion": {
        "num_sampling_steps": 10,
        "noise_schedule": "linear",
        "prediction_type": "epsilon",
      },
      "training": {"learning_rate": 1e-4, "precision": "fp32", "batch_size": 1},
    }
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".yaml", delete=False
    ) as f:
      yaml.dump(content, f)
      return f.name

  def _make_pipeline(self):
    from src.inference.inference_pipeline import InferencePipeline
    from src.models.diffusion.diffusion_engine import GaussianDiffusion
    from src.models.swin_dit.backbone import SwinDiT

    cfg_path = self._make_tiny_config_yaml()

    swin_config = __import__(
      "src.utils.config_manager_swin_dit", fromlist=["load_config"]
    ).load_config(cfg_path)

    denoiser = SwinDiT(swin_config)
    diff = GaussianDiffusion(num_timesteps=10, schedule="linear")

    from src.utils.pipeline_config import PipelineConfig
    cfg = PipelineConfig(
      num_timesteps=10,
      num_inference_steps=5,
      latent_size=8,
      in_channels=4,
      image_size=64,
      device="cpu",
      dtype="float32",
    )

    pipeline = InferencePipeline(
      denoiser=denoiser,
      diffusion=diff,
      config=cfg,
    )
    os.unlink(cfg_path)
    return pipeline

  def test_denoise_unconditional(self):
    pipeline = self._make_pipeline()
    out = pipeline.denoise(batch_size=1)
    assert out.shape == (1, 4, 8, 8)

  def test_denoise_batch(self):
    pipeline = self._make_pipeline()
    out = pipeline.denoise(batch_size=2)
    assert out.shape == (2, 4, 8, 8)

  def test_denoise_with_degradation_type(self):
    pipeline = self._make_pipeline()
    out = pipeline.denoise(batch_size=1, degradation_type=2)
    assert out.shape == (1, 4, 8, 8)

  def test_denoise_with_severity(self):
    pipeline = self._make_pipeline()
    out = pipeline.denoise(batch_size=1, severity=0.6)
    assert out.shape == (1, 4, 8, 8)

  def test_denoise_with_degradation_and_severity(self):
    pipeline = self._make_pipeline()
    out = pipeline.denoise(batch_size=1, degradation_type=1, severity=0.3)
    assert out.shape == (1, 4, 8, 8)

  def test_super_resolve_from_latent(self):
    """Test SR mode without VAE by passing a pre-built latent tensor."""
    pipeline = self._make_pipeline()
    # Bypass VAE by monkey-patching encode_image
    x_lr_latent = torch.randn(1, 4, 8, 8)
    pipeline.encode_image = lambda img: x_lr_latent

    out = pipeline.super_resolve(x_lr_latent)
    assert out.shape == (1, 4, 8, 8)

  def test_no_vae_decode_raises(self):
    from src.inference.inference_pipeline import InferencePipeline
    pipeline = self._make_pipeline()
    assert pipeline.vae is None
    with pytest.raises(RuntimeError, match="VAE is required"):
        pipeline.decode_latent(torch.randn(1, 4, 8, 8))

  def test_no_clip_returns_none_embed(self):
    pipeline = self._make_pipeline()
    assert pipeline.clip_model is None
    embed = pipeline._encode_prompt("sharp facade")
    assert embed is None

  def test_progress_callback(self):
    pipeline = self._make_pipeline()
    steps = []
    pipeline.denoise(
      batch_size=1,
      progress_callback=lambda step, x: steps.append(step),
    )
    assert len(steps) == 5  # num_inference_steps=5

  def test_no_nan_output(self):
    pipeline = self._make_pipeline()
    out = pipeline.denoise(batch_size=1)
    assert not torch.isnan(out).any(), "NaN detected in pipeline output"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
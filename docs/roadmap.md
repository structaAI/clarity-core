# Modular roadmap: Unconditional → Conditional → Integrated

---

### Phase 0: Baseline Swin‑Unet (unconditional restoration)

**Goal:** Validate a robust image restoration pipeline without text conditioning.

- **Architecture:** Build a Swin‑UNet/SCUNet‑style encoder–decoder with skip connections and residual output.
- **Training data:**
  - **Paired LR/noisy → HR/clean** with mixed degradations.
  - **Noise:** Gaussian (varied (\sigma)), Poisson, camera pipeline noise, JPEG artifacts.
  - **Downsampling:** Bicubic/bilinear/nearest; add blur + compression for realism.
- **Losses:**
  - **Distortion:** L1/L2, SSIM.
  - **Perceptual:** LPIPS; optionally lightweight adversarial for SR sharpness.
- **SR head:** Add multi‑scale heads (×2/×4) or a single head with **scale embeddings**.
- **Evaluation:** PSNR, SSIM, LPIPS on synthetic and real noisy datasets; track per‑noise/per‑scale metrics.
- **Ablations:** Depth, Swin window size, skip connection variants, residual scaling.

> **Outcome:** A strong unconditional baseline for denoising and super‑resolution.

---

### Phase 1: Text encoder and conditioning modules (isolated)

**Goal:** Build and validate text processing and conditioning blocks independently.

- **Text encoder:** Integrate **CLIP‑Text** or a compact transformer; freeze or fine‑tune lightly.
- **Cross‑attention:** **Queries from image features**, **keys/values from text embeddings**; implement at multiple resolutions.
- **FiLM modulation:** Generate (\gamma, \beta) from text embeddings; apply to selected blocks.
- **Unit tests:** Synthetic feature maps to validate attention shapes, FiLM broadcasting, token alignment.
- **Stability checks:** Embedding consistency across batch sizes, prompt truncation behavior, positional encoding sanity.

> **Outcome:** Verified text encoder and conditioning modules ready for integration.

---

### Phase 2: Conditional Swin‑Unet (text‑guided restoration)

**Goal:** Integrate text conditioning into Swin‑Unet and test semantic alignment.

- **Integration points:**
  - **Cross‑attention** at bottleneck and late decoder stages.
  - **FiLM** on decoder blocks for style/attribute modulation.
- **Training data:** Noisy/LR → HR pairs with **captions/prompts**; ensure concise, descriptive text.
- **Losses:** Baseline restoration losses + **CLIPScore** alignment; optional attribute classifiers for measurable fidelity.
- **Curriculum:** Start with mild noise/SR; progressively increase degradation severity.
- **Ablations:** Conditioning strength, location (bottleneck vs decoder), type (FiLM vs cross‑attention), prompt length.

> **Outcome:** Text‑guided restoration with measurable semantic fidelity and controlled modulation.

---

### Phase 3: Adaptive gating and noise estimation

**Goal:** Make text conditioning dynamic based on input degradation.

- **Noise estimator:** Small head to predict **corruption level** (noise (\sigma), blur, scale).
- **Gating:** Learned function (e.g., sigmoid) to **scale conditioning strength** per sample.
- **Curriculum:** Low → high noise; mix degradations within batches to avoid mode collapse.
- **Metrics:** Hallucination rate (semantic drift), edit map sparsity, restoration fidelity under high noise.
- **Ablations:** Gated vs fixed conditioning; per‑block gating vs global.

> **Outcome:** Robust, adaptive restoration that avoids over‑conditioning and hallucinations.

---

### Phase 4: Latent Diffusion–UNet (unconditional → conditional)

**Goal:** Build LDM‑UNet for prompt‑guided and blind restoration in latent space.

- **Components:**
  - **VAE (50–100M):** Train/reuse high‑fidelity autoencoder; freeze after convergence.
  - **UNet (600–900M):** Multi‑scale residual blocks with attention; hooks for cross‑attention.
  - **Text encoder (100–150M):** CLIP/Transformer for case 1; optional degradation embeddings for case 2.
- **Training (unconditional first):**
  - **Noise prediction** in latent space with linear/cosine schedule.
  - **Classifier‑free guidance prep:** Drop conditioning with probability (p) to enable guided sampling later.
- **Training (conditional next):**
  - **Case 1:** Text conditioning via cross‑attention; tune guidance scale at inference.
  - **Case 2:** Replace text with **degradation embeddings** or LR latent encoder; no prompt.
- **Losses:** Diffusion MSE on noise; VAE recon (L1/L2 + perceptual) to preserve fidelity.
- **Evaluation:** Distortion (PSNR/SSIM), perception (LPIPS/FID), prompt adherence (case 1), robustness under severe degradations (case 2).

> **Outcome:** A flexible LDM‑UNet that supports both prompt‑guided and blind restoration.

---

### Phase 5: Domain‑specific heads and constraints

**Goal:** Regularize restoration toward domain‑valid outputs.

- **Agriculture:** **Segmentation** and **change detection** heads to preserve structure and temporal consistency.
- **Forensics:** **Face landmarks**, **license plate readability**, **document OCR confidence**.
- **Multi‑task training:** Joint loss with task‑specific weights; freeze/fine‑tune heads as needed.
- **Metrics:** mIoU, F1, NME, readability/OCR scores; measure impact on restoration quality and hallucination.
- **Ablations:** With/without domain heads; per‑domain fine‑tuning vs unified model.

> **Outcome:** Restoration tuned for real‑world constraints and downstream utility.

---

### Phase 6: Provenance, edit tracking, and verifiable outputs

**Goal:** Add edit tracking and reproducible, auditable outputs.

- **Provenance head:** **Edit map** (where changes occurred) + **confidence map**.
- **Manifests:** Model version, prompt, degradation estimates, seeds, guidance scale.
- **Integrity:** Hash output + manifest; anchor to a ledger; store reproducibility metadata.
- **Tooling:** CLI/SDK for verification; batch audit scripts.
- **Metrics:** Unsupported attribute rate, reproducibility rate, drift detection.

> **Outcome:** Trustworthy restoration with verifiable lineage and reproducibility.

---

### Phase 7: Full integration and productization

**Goal:** Combine all modules into a unified pipeline and prepare for deployment.

- **Pipeline:** Restoration (Swin‑Unet) → LDM‑UNet enhancement (optional) → domain heads → provenance.
- **Runtime:** AMP, gradient checkpointing, tiling for large images, ONNX/TensorRT export.
- **API/UI:** Endpoints for unconditional and conditional modes; prompt templates; batch processing.
- **Monitoring:** Drift, confidence, domain metrics; per‑degradation dashboards; prompt adherence (case 1).
- **Docs & reproducibility:** Seeds, configs, manifests; ablation notebooks; evaluation suites.

> **Outcome:** End‑to‑end system ready for real‑world use, benchmarking, and publication.

---

## Model sizes and training data (concise spec)

- **Swin‑UNet (SCUNet‑style):** **30–60M** params; start ~30M, scale depth/window size if metrics plateau.
- **LDM‑UNet:** **600–900M** params; cross‑attention adds modest overhead; use mixed precision + checkpointing.
- **VAE:** **50–100M** params; prioritize recon fidelity; freeze post‑training.
- **Text encoder:** **100–150M** params; CLIP‑Text or compact transformer; freeze or light fine‑tune.

**Data types:**

- **HR clean images:** Diverse scenes (faces, textures, documents, natural/man‑made).
- **Synthetic degradations:** Gaussian/Poisson noise, ISP‑like noise, blur, compression, mixed downsampling.
- **Captions/prompts (case 1):** Concise, descriptive; include attributes relevant to restoration (e.g., “sharp edges”, “fine texture”).
- **Blind restoration (case 2):** Real noisy/LR captures with clean references when available; otherwise realistic synthetic pipelines.

---

## Integration patterns (when components are ready)

- **Cascade:** Swin‑UNet for distortion removal → LDM‑UNet for perceptual enhancement and prompt alignment (case 1).
- **Condition fusion:** Feed Swin‑UNet features/outputs as additional conditioning to LDM‑UNet via encoder tokens or cross‑attention keys/values.
- **Task routing:** Lightweight classifier decides whether to run blind Swin‑UNet only (case 2) or add LDM‑UNet with prompts (case 1).

---

## Compute and training tips

- **Hardware:** Swin‑UNet: 1–2×24GB GPUs; LDM‑UNet: 4–8×24–80GB GPUs.
- **Stability:** EMA weights, cosine LR, gradient clipping; validate on real noisy/LR splits.
- **Efficiency:** Mixed precision, gradient checkpointing, micro‑batching; distributed data‑parallel.
- **Evaluation cadence:** Per‑noise/per‑scale metrics; prompt adherence (case 1); hallucination audits.

---

## Modular roadmap: Unconditional → Conditional → Integrated

---

### Phase 0: Baseline Swin‑Unet (unconditional restoration)

**Goal:** Validate a robust image restoration pipeline without text conditioning.

- **Architecture:** Build a Swin‑UNet/SCUNet‑style encoder–decoder with skip connections and residual output.
- **Training data:**
  - **Paired LR/noisy → HR/clean** with mixed degradations.
  - **Noise:** Gaussian (varied (\sigma)), Poisson, camera pipeline noise, JPEG artifacts.
  - **Downsampling:** Bicubic/bilinear/nearest; add blur + compression for realism.
- **Losses:**
  - **Distortion:** L1/L2, SSIM.
  - **Perceptual:** LPIPS; optionally lightweight adversarial for SR sharpness.
- **SR head:** Add multi‑scale heads (×2/×4) or a single head with **scale embeddings**.
- **Evaluation:** PSNR, SSIM, LPIPS on synthetic and real noisy datasets; track per‑noise/per‑scale metrics.
- **Ablations:** Depth, Swin window size, skip connection variants, residual scaling.

> **Outcome:** A strong unconditional baseline for denoising and super‑resolution.

---

### Phase 1: Text encoder and conditioning modules (isolated)

**Goal:** Build and validate text processing and conditioning blocks independently.

- **Text encoder:** Integrate **CLIP‑Text** or a compact transformer; freeze or fine‑tune lightly.
- **Cross‑attention:** **Queries from image features**, **keys/values from text embeddings**; implement at multiple resolutions.
- **FiLM modulation:** Generate (\gamma, \beta) from text embeddings; apply to selected blocks.
- **Unit tests:** Synthetic feature maps to validate attention shapes, FiLM broadcasting, token alignment.
- **Stability checks:** Embedding consistency across batch sizes, prompt truncation behavior, positional encoding sanity.

> **Outcome:** Verified text encoder and conditioning modules ready for integration.

---

### Phase 2: Conditional Swin‑Unet (text‑guided restoration)

**Goal:** Integrate text conditioning into Swin‑Unet and test semantic alignment.

- **Integration points:**
  - **Cross‑attention** at bottleneck and late decoder stages.
  - **FiLM** on decoder blocks for style/attribute modulation.
- **Training data:** Noisy/LR → HR pairs with **captions/prompts**; ensure concise, descriptive text.
- **Losses:** Baseline restoration losses + **CLIPScore** alignment; optional attribute classifiers for measurable fidelity.
- **Curriculum:** Start with mild noise/SR; progressively increase degradation severity.
- **Ablations:** Conditioning strength, location (bottleneck vs decoder), type (FiLM vs cross‑attention), prompt length.

> **Outcome:** Text‑guided restoration with measurable semantic fidelity and controlled modulation.

---

### Phase 3: Adaptive gating and noise estimation

**Goal:** Make text conditioning dynamic based on input degradation.

- **Noise estimator:** Small head to predict **corruption level** (noise (\sigma), blur, scale).
- **Gating:** Learned function (e.g., sigmoid) to **scale conditioning strength** per sample.
- **Curriculum:** Low → high noise; mix degradations within batches to avoid mode collapse.
- **Metrics:** Hallucination rate (semantic drift), edit map sparsity, restoration fidelity under high noise.
- **Ablations:** Gated vs fixed conditioning; per‑block gating vs global.

> **Outcome:** Robust, adaptive restoration that avoids over‑conditioning and hallucinations.

---

### Phase 4: Latent Diffusion–UNet (unconditional → conditional)

**Goal:** Build LDM‑UNet for prompt‑guided and blind restoration in latent space.

- **Components:**
  - **VAE (50–100M):** Train/reuse high‑fidelity autoencoder; freeze after convergence.
  - **UNet (600–900M):** Multi‑scale residual blocks with attention; hooks for cross‑attention.
  - **Text encoder (100–150M):** CLIP/Transformer for case 1; optional degradation embeddings for case 2.
- **Training (unconditional first):**
  - **Noise prediction** in latent space with linear/cosine schedule.
  - **Classifier‑free guidance prep:** Drop conditioning with probability (p) to enable guided sampling later.
- **Training (conditional next):**
  - **Case 1:** Text conditioning via cross‑attention; tune guidance scale at inference.
  - **Case 2:** Replace text with **degradation embeddings** or LR latent encoder; no prompt.
- **Losses:** Diffusion MSE on noise; VAE recon (L1/L2 + perceptual) to preserve fidelity.
- **Evaluation:** Distortion (PSNR/SSIM), perception (LPIPS/FID), prompt adherence (case 1), robustness under severe degradations (case 2).

> **Outcome:** A flexible LDM‑UNet that supports both prompt‑guided and blind restoration.

---

### Phase 5: Domain‑specific heads and constraints

**Goal:** Regularize restoration toward domain‑valid outputs.

- **Agriculture:** **Segmentation** and **change detection** heads to preserve structure and temporal consistency.
- **Forensics:** **Face landmarks**, **license plate readability**, **document OCR confidence**.
- **Multi‑task training:** Joint loss with task‑specific weights; freeze/fine‑tune heads as needed.
- **Metrics:** mIoU, F1, NME, readability/OCR scores; measure impact on restoration quality and hallucination.
- **Ablations:** With/without domain heads; per‑domain fine‑tuning vs unified model.

> **Outcome:** Restoration tuned for real‑world constraints and downstream utility.

---

### Phase 6: Provenance, edit tracking, and verifiable outputs

**Goal:** Add edit tracking and reproducible, auditable outputs.

- **Provenance head:** **Edit map** (where changes occurred) + **confidence map**.
- **Manifests:** Model version, prompt, degradation estimates, seeds, guidance scale.
- **Integrity:** Hash output + manifest; anchor to a ledger; store reproducibility metadata.
- **Tooling:** CLI/SDK for verification; batch audit scripts.
- **Metrics:** Unsupported attribute rate, reproducibility rate, drift detection.

> **Outcome:** Trustworthy restoration with verifiable lineage and reproducibility.

---

### Phase 7: Full integration and productization

**Goal:** Combine all modules into a unified pipeline and prepare for deployment.

- **Pipeline:** Restoration (Swin‑Unet) → LDM‑UNet enhancement (optional) → domain heads → provenance.
- **Runtime:** AMP, gradient checkpointing, tiling for large images, ONNX/TensorRT export.
- **API/UI:** Endpoints for unconditional and conditional modes; prompt templates; batch processing.
- **Monitoring:** Drift, confidence, domain metrics; per‑degradation dashboards; prompt adherence (case 1).
- **Docs & reproducibility:** Seeds, configs, manifests; ablation notebooks; evaluation suites.

> **Outcome:** End‑to‑end system ready for real‑world use, benchmarking, and publication.

---

## Model sizes and training data (concise spec)

- **Swin‑UNet (SCUNet‑style):** **30–60M** params; start ~30M, scale depth/window size if metrics plateau.
- **LDM‑UNet:** **600–900M** params; cross‑attention adds modest overhead; use mixed precision + checkpointing.
- **VAE:** **50–100M** params; prioritize recon fidelity; freeze post‑training.
- **Text encoder:** **100–150M** params; CLIP‑Text or compact transformer; freeze or light fine‑tune.

**Data types:**

- **HR clean images:** Diverse scenes (faces, textures, documents, natural/man‑made).
- **Synthetic degradations:** Gaussian/Poisson noise, ISP‑like noise, blur, compression, mixed downsampling.
- **Captions/prompts (case 1):** Concise, descriptive; include attributes relevant to restoration (e.g., “sharp edges”, “fine texture”).
- **Blind restoration (case 2):** Real noisy/LR captures with clean references when available; otherwise realistic synthetic pipelines.

---

## Integration patterns (when components are ready)

- **Cascade:** Swin‑UNet for distortion removal → LDM‑UNet for perceptual enhancement and prompt alignment (case 1).
- **Condition fusion:** Feed Swin‑UNet features/outputs as additional conditioning to LDM‑UNet via encoder tokens or cross‑attention keys/values.
- **Task routing:** Lightweight classifier decides whether to run blind Swin‑UNet only (case 2) or add LDM‑UNet with prompts (case 1).

---

## Compute and training tips

- **Hardware:** Swin‑UNet: 1–2×24GB GPUs; LDM‑UNet: 4–8×24–80GB GPUs.
- **Stability:** EMA weights, cosine LR, gradient clipping; validate on real noisy/LR splits.
- **Efficiency:** Mixed precision, gradient checkpointing, micro‑batching; distributed data‑parallel.
- **Evaluation cadence:** Per‑noise/per‑scale metrics; prompt adherence (case 1); hallucination audits.

---

## Modular roadmap: Unconditional → Conditional → Integrated

---

### Phase 0: Baseline Swin‑Unet (unconditional restoration)

**Goal:** Validate a robust image restoration pipeline without text conditioning.

- **Architecture:** Build a Swin‑UNet/SCUNet‑style encoder–decoder with skip connections and residual output.
- **Training data:**
  - **Paired LR/noisy → HR/clean** with mixed degradations.
  - **Noise:** Gaussian (varied (\sigma)), Poisson, camera pipeline noise, JPEG artifacts.
  - **Downsampling:** Bicubic/bilinear/nearest; add blur + compression for realism.
- **Losses:**
  - **Distortion:** L1/L2, SSIM.
  - **Perceptual:** LPIPS; optionally lightweight adversarial for SR sharpness.
- **SR head:** Add multi‑scale heads (×2/×4) or a single head with **scale embeddings**.
- **Evaluation:** PSNR, SSIM, LPIPS on synthetic and real noisy datasets; track per‑noise/per‑scale metrics.
- **Ablations:** Depth, Swin window size, skip connection variants, residual scaling.

> **Outcome:** A strong unconditional baseline for denoising and super‑resolution.

---

### Phase 1: Text encoder and conditioning modules (isolated)

**Goal:** Build and validate text processing and conditioning blocks independently.

- **Text encoder:** Integrate **CLIP‑Text** or a compact transformer; freeze or fine‑tune lightly.
- **Cross‑attention:** **Queries from image features**, **keys/values from text embeddings**; implement at multiple resolutions.
- **FiLM modulation:** Generate (\gamma, \beta) from text embeddings; apply to selected blocks.
- **Unit tests:** Synthetic feature maps to validate attention shapes, FiLM broadcasting, token alignment.
- **Stability checks:** Embedding consistency across batch sizes, prompt truncation behavior, positional encoding sanity.

> **Outcome:** Verified text encoder and conditioning modules ready for integration.

---

### Phase 2: Conditional Swin‑Unet (text‑guided restoration)

**Goal:** Integrate text conditioning into Swin‑Unet and test semantic alignment.

- **Integration points:**
  - **Cross‑attention** at bottleneck and late decoder stages.
  - **FiLM** on decoder blocks for style/attribute modulation.
- **Training data:** Noisy/LR → HR pairs with **captions/prompts**; ensure concise, descriptive text.
- **Losses:** Baseline restoration losses + **CLIPScore** alignment; optional attribute classifiers for measurable fidelity.
- **Curriculum:** Start with mild noise/SR; progressively increase degradation severity.
- **Ablations:** Conditioning strength, location (bottleneck vs decoder), type (FiLM vs cross‑attention), prompt length.

> **Outcome:** Text‑guided restoration with measurable semantic fidelity and controlled modulation.

---

### Phase 3: Adaptive gating and noise estimation

**Goal:** Make text conditioning dynamic based on input degradation.

- **Noise estimator:** Small head to predict **corruption level** (noise (\sigma), blur, scale).
- **Gating:** Learned function (e.g., sigmoid) to **scale conditioning strength** per sample.
- **Curriculum:** Low → high noise; mix degradations within batches to avoid mode collapse.
- **Metrics:** Hallucination rate (semantic drift), edit map sparsity, restoration fidelity under high noise.
- **Ablations:** Gated vs fixed conditioning; per‑block gating vs global.

> **Outcome:** Robust, adaptive restoration that avoids over‑conditioning and hallucinations.

---

### Phase 4: Latent Diffusion–UNet (unconditional → conditional)

**Goal:** Build LDM‑UNet for prompt‑guided and blind restoration in latent space.

- **Components:**
  - **VAE (50–100M):** Train/reuse high‑fidelity autoencoder; freeze after convergence.
  - **UNet (600–900M):** Multi‑scale residual blocks with attention; hooks for cross‑attention.
  - **Text encoder (100–150M):** CLIP/Transformer for case 1; optional degradation embeddings for case 2.
- **Training (unconditional first):**
  - **Noise prediction** in latent space with linear/cosine schedule.
  - **Classifier‑free guidance prep:** Drop conditioning with probability (p) to enable guided sampling later.
- **Training (conditional next):**
  - **Case 1:** Text conditioning via cross‑attention; tune guidance scale at inference.
  - **Case 2:** Replace text with **degradation embeddings** or LR latent encoder; no prompt.
- **Losses:** Diffusion MSE on noise; VAE recon (L1/L2 + perceptual) to preserve fidelity.
- **Evaluation:** Distortion (PSNR/SSIM), perception (LPIPS/FID), prompt adherence (case 1), robustness under severe degradations (case 2).

> **Outcome:** A flexible LDM‑UNet that supports both prompt‑guided and blind restoration.

---

### Phase 5: Domain‑specific heads and constraints

**Goal:** Regularize restoration toward domain‑valid outputs.

- **Agriculture:** **Segmentation** and **change detection** heads to preserve structure and temporal consistency.
- **Forensics:** **Face landmarks**, **license plate readability**, **document OCR confidence**.
- **Multi‑task training:** Joint loss with task‑specific weights; freeze/fine‑tune heads as needed.
- **Metrics:** mIoU, F1, NME, readability/OCR scores; measure impact on restoration quality and hallucination.
- **Ablations:** With/without domain heads; per‑domain fine‑tuning vs unified model.

> **Outcome:** Restoration tuned for real‑world constraints and downstream utility.

---

### Phase 6: Provenance, edit tracking, and verifiable outputs

**Goal:** Add edit tracking and reproducible, auditable outputs.

- **Provenance head:** **Edit map** (where changes occurred) + **confidence map**.
- **Manifests:** Model version, prompt, degradation estimates, seeds, guidance scale.
- **Integrity:** Hash output + manifest; anchor to a ledger; store reproducibility metadata.
- **Tooling:** CLI/SDK for verification; batch audit scripts.
- **Metrics:** Unsupported attribute rate, reproducibility rate, drift detection.

> **Outcome:** Trustworthy restoration with verifiable lineage and reproducibility.

---

### Phase 7: Full integration and productization

**Goal:** Combine all modules into a unified pipeline and prepare for deployment.

- **Pipeline:** Restoration (Swin‑Unet) → LDM‑UNet enhancement (optional) → domain heads → provenance.
- **Runtime:** AMP, gradient checkpointing, tiling for large images, ONNX/TensorRT export.
- **API/UI:** Endpoints for unconditional and conditional modes; prompt templates; batch processing.
- **Monitoring:** Drift, confidence, domain metrics; per‑degradation dashboards; prompt adherence (case 1).
- **Docs & reproducibility:** Seeds, configs, manifests; ablation notebooks; evaluation suites.

> **Outcome:** End‑to‑end system ready for real‑world use, benchmarking, and publication.

---

## Model sizes and training data (concise spec)

- **Swin‑UNet (SCUNet‑style):** **30–60M** params; start ~30M, scale depth/window size if metrics plateau.
- **LDM‑UNet:** **600–900M** params; cross‑attention adds modest overhead; use mixed precision + checkpointing.
- **VAE:** **50–100M** params; prioritize recon fidelity; freeze post‑training.
- **Text encoder:** **100–150M** params; CLIP‑Text or compact transformer; freeze or light fine‑tune.

**Data types:**

- **HR clean images:** Diverse scenes (faces, textures, documents, natural/man‑made).
- **Synthetic degradations:** Gaussian/Poisson noise, ISP‑like noise, blur, compression, mixed downsampling.
- **Captions/prompts (case 1):** Concise, descriptive; include attributes relevant to restoration (e.g., “sharp edges”, “fine texture”).
- **Blind restoration (case 2):** Real noisy/LR captures with clean references when available; otherwise realistic synthetic pipelines.

---

## Integration patterns (when components are ready)

- **Cascade:** Swin‑UNet for distortion removal → LDM‑UNet for perceptual enhancement and prompt alignment (case 1).
- **Condition fusion:** Feed Swin‑UNet features/outputs as additional conditioning to LDM‑UNet via encoder tokens or cross‑attention keys/values.
- **Task routing:** Lightweight classifier decides whether to run blind Swin‑UNet only (case 2) or add LDM‑UNet with prompts (case 1).

---

## Compute and training tips

- **Hardware:** Swin‑UNet: 1–2×24GB GPUs; LDM‑UNet: 4–8×24–80GB GPUs.
- **Stability:** EMA weights, cosine LR, gradient clipping; validate on real noisy/LR splits.
- **Efficiency:** Mixed precision, gradient checkpointing, micro‑batching; distributed data‑parallel.
- **Evaluation cadence:** Per‑noise/per‑scale metrics; prompt adherence (case 1); hallucination audits.

---

## Modular roadmap: Unconditional → Conditional → Integrated

---

### Phase 0: Baseline Swin‑Unet (unconditional restoration)

**Goal:** Validate a robust image restoration pipeline without text conditioning.

- **Architecture:** Build a Swin‑UNet/SCUNet‑style encoder–decoder with skip connections and residual output.
- **Training data:**
  - **Paired LR/noisy → HR/clean** with mixed degradations.
  - **Noise:** Gaussian (varied (\sigma)), Poisson, camera pipeline noise, JPEG artifacts.
  - **Downsampling:** Bicubic/bilinear/nearest; add blur + compression for realism.
- **Losses:**
  - **Distortion:** L1/L2, SSIM.
  - **Perceptual:** LPIPS; optionally lightweight adversarial for SR sharpness.
- **SR head:** Add multi‑scale heads (×2/×4) or a single head with **scale embeddings**.
- **Evaluation:** PSNR, SSIM, LPIPS on synthetic and real noisy datasets; track per‑noise/per‑scale metrics.
- **Ablations:** Depth, Swin window size, skip connection variants, residual scaling.

> **Outcome:** A strong unconditional baseline for denoising and super‑resolution.

---

### Phase 1: Text encoder and conditioning modules (isolated)

**Goal:** Build and validate text processing and conditioning blocks independently.

- **Text encoder:** Integrate **CLIP‑Text** or a compact transformer; freeze or fine‑tune lightly.
- **Cross‑attention:** **Queries from image features**, **keys/values from text embeddings**; implement at multiple resolutions.
- **FiLM modulation:** Generate (\gamma, \beta) from text embeddings; apply to selected blocks.
- **Unit tests:** Synthetic feature maps to validate attention shapes, FiLM broadcasting, token alignment.
- **Stability checks:** Embedding consistency across batch sizes, prompt truncation behavior, positional encoding sanity.

> **Outcome:** Verified text encoder and conditioning modules ready for integration.

---

### Phase 2: Conditional Swin‑Unet (text‑guided restoration)

**Goal:** Integrate text conditioning into Swin‑Unet and test semantic alignment.

- **Integration points:**
  - **Cross‑attention** at bottleneck and late decoder stages.
  - **FiLM** on decoder blocks for style/attribute modulation.
- **Training data:** Noisy/LR → HR pairs with **captions/prompts**; ensure concise, descriptive text.
- **Losses:** Baseline restoration losses + **CLIPScore** alignment; optional attribute classifiers for measurable fidelity.
- **Curriculum:** Start with mild noise/SR; progressively increase degradation severity.
- **Ablations:** Conditioning strength, location (bottleneck vs decoder), type (FiLM vs cross‑attention), prompt length.

> **Outcome:** Text‑guided restoration with measurable semantic fidelity and controlled modulation.

---

### Phase 3: Adaptive gating and noise estimation

**Goal:** Make text conditioning dynamic based on input degradation.

- **Noise estimator:** Small head to predict **corruption level** (noise (\sigma), blur, scale).
- **Gating:** Learned function (e.g., sigmoid) to **scale conditioning strength** per sample.
- **Curriculum:** Low → high noise; mix degradations within batches to avoid mode collapse.
- **Metrics:** Hallucination rate (semantic drift), edit map sparsity, restoration fidelity under high noise.
- **Ablations:** Gated vs fixed conditioning; per‑block gating vs global.

> **Outcome:** Robust, adaptive restoration that avoids over‑conditioning and hallucinations.

---

### Phase 4: Latent Diffusion–UNet (unconditional → conditional)

**Goal:** Build LDM‑UNet for prompt‑guided and blind restoration in latent space.

- **Components:**
  - **VAE (50–100M):** Train/reuse high‑fidelity autoencoder; freeze after convergence.
  - **UNet (600–900M):** Multi‑scale residual blocks with attention; hooks for cross‑attention.
  - **Text encoder (100–150M):** CLIP/Transformer for case 1; optional degradation embeddings for case 2.
- **Training (unconditional first):**
  - **Noise prediction** in latent space with linear/cosine schedule.
  - **Classifier‑free guidance prep:** Drop conditioning with probability (p) to enable guided sampling later.
- **Training (conditional next):**
  - **Case 1:** Text conditioning via cross‑attention; tune guidance scale at inference.
  - **Case 2:** Replace text with **degradation embeddings** or LR latent encoder; no prompt.
- **Losses:** Diffusion MSE on noise; VAE recon (L1/L2 + perceptual) to preserve fidelity.
- **Evaluation:** Distortion (PSNR/SSIM), perception (LPIPS/FID), prompt adherence (case 1), robustness under severe degradations (case 2).

> **Outcome:** A flexible LDM‑UNet that supports both prompt‑guided and blind restoration.

---

### Phase 5: Domain‑specific heads and constraints

**Goal:** Regularize restoration toward domain‑valid outputs.

- **Agriculture:** **Segmentation** and **change detection** heads to preserve structure and temporal consistency.
- **Forensics:** **Face landmarks**, **license plate readability**, **document OCR confidence**.
- **Multi‑task training:** Joint loss with task‑specific weights; freeze/fine‑tune heads as needed.
- **Metrics:** mIoU, F1, NME, readability/OCR scores; measure impact on restoration quality and hallucination.
- **Ablations:** With/without domain heads; per‑domain fine‑tuning vs unified model.

> **Outcome:** Restoration tuned for real‑world constraints and downstream utility.

---

### Phase 6: Provenance, edit tracking, and verifiable outputs

**Goal:** Add edit tracking and reproducible, auditable outputs.

- **Provenance head:** **Edit map** (where changes occurred) + **confidence map**.
- **Manifests:** Model version, prompt, degradation estimates, seeds, guidance scale.
- **Integrity:** Hash output + manifest; anchor to a ledger; store reproducibility metadata.
- **Tooling:** CLI/SDK for verification; batch audit scripts.
- **Metrics:** Unsupported attribute rate, reproducibility rate, drift detection.

> **Outcome:** Trustworthy restoration with verifiable lineage and reproducibility.

---

### Phase 7: Full integration and productization

**Goal:** Combine all modules into a unified pipeline and prepare for deployment.

- **Pipeline:** Restoration (Swin‑Unet) → LDM‑UNet enhancement (optional) → domain heads → provenance.
- **Runtime:** AMP, gradient checkpointing, tiling for large images, ONNX/TensorRT export.
- **API/UI:** Endpoints for unconditional and conditional modes; prompt templates; batch processing.
- **Monitoring:** Drift, confidence, domain metrics; per‑degradation dashboards; prompt adherence (case 1).
- **Docs & reproducibility:** Seeds, configs, manifests; ablation notebooks; evaluation suites.

> **Outcome:** End‑to‑end system ready for real‑world use, benchmarking, and publication.

---

## Model sizes and training data (concise spec)

- **Swin‑UNet (SCUNet‑style):** **30–60M** params; start ~30M, scale depth/window size if metrics plateau.
- **LDM‑UNet:** **600–900M** params; cross‑attention adds modest overhead; use mixed precision + checkpointing.
- **VAE:** **50–100M** params; prioritize recon fidelity; freeze post‑training.
- **Text encoder:** **100–150M** params; CLIP‑Text or compact transformer; freeze or light fine‑tune.

**Data types:**

- **HR clean images:** Diverse scenes (faces, textures, documents, natural/man‑made).
- **Synthetic degradations:** Gaussian/Poisson noise, ISP‑like noise, blur, compression, mixed downsampling.
- **Captions/prompts (case 1):** Concise, descriptive; include attributes relevant to restoration (e.g., “sharp edges”, “fine texture”).
- **Blind restoration (case 2):** Real noisy/LR captures with clean references when available; otherwise realistic synthetic pipelines.

---

## Integration patterns (when components are ready)

- **Cascade:** Swin‑UNet for distortion removal → LDM‑UNet for perceptual enhancement and prompt alignment (case 1).
- **Condition fusion:** Feed Swin‑UNet features/outputs as additional conditioning to LDM‑UNet via encoder tokens or cross‑attention keys/values.
- **Task routing:** Lightweight classifier decides whether to run blind Swin‑UNet only (case 2) or add LDM‑UNet with prompts (case 1).

---

## Compute and training tips

- **Hardware:** Swin‑UNet: 1–2×24GB GPUs; LDM‑UNet: 4–8×24–80GB GPUs.
- **Stability:** EMA weights, cosine LR, gradient clipping; validate on real noisy/LR splits.
- **Efficiency:** Mixed precision, gradient checkpointing, micro‑batching; distributed data‑parallel.
- **Evaluation cadence:** Per‑noise/per‑scale metrics; prompt adherence (case 1); hallucination audits.

---

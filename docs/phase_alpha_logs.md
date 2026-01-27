# Logging Progress

## Phase $\alpha$.0 (Reading Research Papers)

---

## Phase $\alpha$.1 (Building the Variational AutoEncoder)

---

## Phase $\alpha$.2 (Working on the Swin-DiT backbone)

### Step-1: Patchification (Latents to Tokens)

Converted the $64 \times 64 \times 4$ latents into a sequence of patches. For research on scaling, this drastically reduces the sequence length compared to pixel-space models.

### Step-2: Timestep Conditioning

Implemented a Timestep Embedder (MLP) to project the diffusion noise level $t$ into a high-dimensional vector space. This vector must be injected into every block so the Swin-Transformer knows the intensity of noise it is handling.

### Step-3: Hierarchical Swin Blocks

Built the core transformer blocks using Shifted Window Attention. This allows the model to capture local textures (like fine edges in an image) while maintaining the efficiency of non-global attention.

### Step-4: Pseudo-Shifted Window Attention

This is the specific novelty of our research. We have implemented the parallel high-frequency bridge that specifically targets restoration details that the VAE might have softened.

### Step-5: Latent Reconstruction Head

A final linear/convolutional layer that projects the transformer tokens back into the $64 \times 64 \times 4$ latent shape for the VAE decoder to eventually read.

---

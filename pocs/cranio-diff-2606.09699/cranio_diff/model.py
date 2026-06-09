"""
cranio_diff/model.py

Cranio-Diff: ControlNet-guided Stable Diffusion for skull-to-face reconstruction.
Paper: arxiv.org/abs/2606.09699

Architecture (Figure 3):
  - Frozen VAE encoder/decoder  (Stable Diffusion v1.5)
  - Frozen CLIP text encoder
  - Trainable ControlNet branch (skull conditioning)
  - Trainable denoising UNet

Key equations:
  F_s^(0) = Z_in(X_s)                         [Eq.1]  zero-conv input
  F_s     = E_ctrl(F_s^(0) ⊕ F_unet)          [Eq.2]  trainable encoder copy
  C_s     = Z_out(F_s)                          [Eq.3]  zero-conv output
  z_t     = √ᾱ_t·z_0 + √(1-ᾱ_t)·ε             [Eq.7]  forward diffusion
"""

from __future__ import annotations
import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


# ── Biometric text prompt builder ────────────────────────────────────────────

AGE_GROUPS = {25: "young adult", 45: "middle-aged adult", 65: "elderly adult"}
BMI_LABELS = {-10: "slim build", 0: "average build", 10: "heavy build"}

def build_biometric_prompt(
    age: int = 45,
    gender: str = "male",
    orientation: str = "frontal",
    bmi_delta: int = 0,
) -> str:
    """
    Constructs standardized biometric text prompt (Section 4.3).

    Args:
        age:         25 / 45 / 65
        gender:      "male" / "female"
        orientation: "frontal" / "lateral"
        bmi_delta:   -10 / 0 / +10 (percent)

    Returns:
        Text string fed to frozen CLIP text encoder.

    Example:
        >>> build_biometric_prompt(45, "male", "frontal", 0)
        'A middle-aged adult male, frontal view, average build, photorealistic face'
    """
    age_desc = AGE_GROUPS.get(age, "adult")
    bmi_desc = BMI_LABELS.get(bmi_delta, "average build")
    return (
        f"A {age_desc} {gender}, {orientation} view, {bmi_desc}, "
        "photorealistic face, high detail, craniofacial reconstruction"
    )


# ── Zero Convolution (ControlNet core) ───────────────────────────────────────

class ZeroConv2d(nn.Module):
    """
    Zero-initialized convolution layer (ControlNet).
    At init: output is exactly zero → no perturbation to pretrained UNet.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Cranio-Diff Training Wrapper ─────────────────────────────────────────────

class CranioDiff(nn.Module):
    """
    Trainable wrapper around SD v1.5 + ControlNet for skull-conditioned
    facial reconstruction.

    Frozen:   VAE encoder, VAE decoder, CLIP text encoder
    Trainable: ControlNet branch, denoising UNet
    """

    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    # Paper uses Realistic Vision v5.1 (fine-tuned SD v1.5)
    # For PoC, standard SD v1.5 is used; swap MODEL_ID to
    # "SG161222/Realistic_Vision_V5.1_noVAE" for full fidelity.

    def __init__(
        self,
        controlnet_model_id: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # ── Load frozen components ──────────────────────────────────────────
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            self.MODEL_ID, subfolder="vae"
        ).to(device, dtype)
        self.vae.requires_grad_(False)

        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            self.MODEL_ID, subfolder="text_encoder"
        ).to(device, dtype)
        self.text_encoder.requires_grad_(False)

        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            self.MODEL_ID, subfolder="tokenizer"
        )

        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            self.MODEL_ID, subfolder="scheduler"
        )

        # ── Load trainable components ───────────────────────────────────────
        if controlnet_model_id:
            self.controlnet: ControlNetModel = ControlNetModel.from_pretrained(
                controlnet_model_id
            ).to(device, dtype)
        else:
            # Initialize ControlNet from UNet (standard ControlNet init)
            self.controlnet = ControlNetModel.from_unet(
                UNet2DConditionModel.from_pretrained(self.MODEL_ID, subfolder="unet")
            ).to(device, dtype)

        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.MODEL_ID, subfolder="unet"
        ).to(device, dtype)

        # Only ControlNet + UNet are trainable
        self.controlnet.requires_grad_(True)
        self.unet.requires_grad_(True)

    # ── Text encoding ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """
        Encodes biometric text prompts → τ ∈ R^{L×d}  (Eq. 4)
        """
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens).last_hidden_state

    # ── VAE encoding ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Maps face image x_0 → latent z_0  (Section 4.4)
        images: (B, 3, H, W) in [-1, 1]
        """
        latents = self.vae.encode(images.to(self.dtype)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent ẑ_0 → synthesized face image x̂_0  (Section 4.4)
        """
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents.to(self.dtype)).sample

    # ── Forward diffusion ─────────────────────────────────────────────────────

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: z_t = √ᾱ_t·z_0 + √(1-ᾱ_t)·ε   (Eq. 7)
        """
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        face_images: torch.Tensor,   # (B, 3, 512, 512) real faces in [-1,1]
        skull_images: torch.Tensor,  # (B, 3, 512, 512) skull X-rays in [-1,1]
        prompts: list[str],          # biometric text prompts
    ) -> torch.Tensor:
        """
        One training step. Returns diffusion loss L_diff (Eq. 8).

        L_diff = E[||ε - ε_θ(z_t, t, C_s, τ)||²]

        Full loss = L_diff + λ1·L_LPIPS + λ2·L_id
        is computed in train.py using decoded images.
        """
        B = face_images.shape[0]

        # 1. Encode face → latent z_0
        z0 = self.encode_image(face_images)

        # 2. Sample noise ε ~ N(0, I) and timestep t
        noise = torch.randn_like(z0)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # 3. Forward diffusion → z_t  (Eq. 7)
        z_t = self.add_noise(z0, noise, timesteps)

        # 4. Encode text prompts → τ  (Eq. 4)
        text_embeddings = self.encode_text(prompts)

        # 5. ControlNet skull conditioning → C_s  (Eq. 1-3)
        #    Z_in → E_ctrl(· ⊕ F_unet) → Z_out
        down_block_res, mid_block_res = self.controlnet(
            z_t,
            timesteps,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=skull_images.to(self.dtype),
            return_dict=False,
        )

        # 6. Denoising UNet: ε_θ(z_t, t, C_s, τ)  (Eq. 8)
        noise_pred = self.unet(
            z_t,
            timesteps,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res,
            mid_block_additional_residual=mid_block_res,
        ).sample

        # 7. Diffusion loss: ||ε - ε_θ||²
        return torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        skull_image: torch.Tensor,   # (1, 3, 512, 512)
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Full inference: skull X-ray + biometric prompt → reconstructed face.
        Returns: (1, 3, 512, 512) tensor in [-1, 1].
        """
        pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(self.device)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        result = pipeline(
            prompt=prompt,
            image=skull_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images[0]

        return result

# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from ..normalization import NormType
from .audio_vae import AudioDecoder, AudioEncoder
from .causality_axis import CausalityAxis
from .vocoder import Vocoder


def _audio_vae_sections(config: dict) -> tuple[dict, dict, dict, int | None]:
    audio_vae_cfg = config.get("audio_vae", {})
    model_params = audio_vae_cfg.get("model", {}).get("params", {})
    ddconfig = model_params.get("ddconfig", {})
    preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
    stft_cfg = preprocessing_cfg.get("stft", {})
    mel_bins = (
        ddconfig.get("mel_bins")
        or preprocessing_cfg.get("mel", {}).get("n_mel_channels")
        or audio_vae_cfg.get("variables", {}).get("mel_bins")
    )
    return model_params, ddconfig, stft_cfg, mel_bins


class VocoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> Vocoder:
        config = config.get("vocoder", {})
        return Vocoder(
            resblock_kernel_sizes=config.get("resblock_kernel_sizes", [3, 7, 11]),
            upsample_rates=config.get("upsample_rates", [6, 5, 2, 2, 2]),
            upsample_kernel_sizes=config.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
            resblock_dilation_sizes=config.get(
                "resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            ),
            upsample_initial_channel=config.get("upsample_initial_channel", 1024),
            stereo=config.get("stereo", True),
            resblock=config.get("resblock", "1"),
            output_sample_rate=config.get("output_sample_rate", 24000),
        )


class AudioDecoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> AudioDecoder:
        model_params, ddconfig, stft_cfg, mel_bins = _audio_vae_sections(config)
        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        is_causal = stft_cfg.get("causal", True)

        return AudioDecoder(
            ch=ddconfig.get("ch", 128),
            out_ch=ddconfig.get("out_ch", 2),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=ddconfig.get("attn_resolutions", {8, 16, 32}),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            dropout=ddconfig.get("dropout", 0.0),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )


class AudioEncoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> AudioEncoder:
        model_params, ddconfig, stft_cfg, mel_bins = _audio_vae_sections(config)
        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        n_fft = stft_cfg.get("filter_length", 1024)
        is_causal = stft_cfg.get("causal", True)
        if mel_bins is None:
            raise ValueError("LTX-2 audio VAE config does not define the mel-bin count.")

        # The native encoder implements only vanilla attention.
        attn_type = ddconfig.get("attn_type", "vanilla")
        if attn_type != "vanilla":
            raise ValueError(
                f"LTX-2 native audio encoder supports attn_type='vanilla'; got {attn_type!r}."
            )

        return AudioEncoder(
            ch=ddconfig.get("ch", 128),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=ddconfig.get("attn_resolutions", {8, 16, 32}),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            double_z=ddconfig.get("double_z", True),
            dropout=ddconfig.get("dropout", 0.0),
            resamp_with_conv=ddconfig.get("resamp_with_conv", True),
            in_channels=ddconfig.get("in_channels", 2),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )

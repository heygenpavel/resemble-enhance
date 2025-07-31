import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from torch import nn, Tensor


@dataclass
class HParams:
    wav_rate: int = 44_100
    n_fft: int = 2048
    win_size: int = 2048
    hop_size: int = 420
    num_mels: int = 128
    stft_magnitude_min: float = 1e-4
    preemphasis: float = 0.97


class MelSpectrogram(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=hp.wav_rate,
            n_fft=hp.n_fft,
            win_length=hp.win_size,
            hop_length=hp.hop_size,
            f_min=0,
            f_max=hp.wav_rate // 2,
            n_mels=hp.num_mels,
            power=1,
            normalized=False,
            pad_mode="constant",
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("stft_magnitude_min", torch.tensor([hp.stft_magnitude_min]))
        self.min_level_db = 20 * torch.log10(self.stft_magnitude_min)
        self.preemphasis = hp.preemphasis
        self.hop_size = hp.hop_size

    def forward(self, wav: Tensor) -> Tensor:
        device = wav.device
        if self.preemphasis > 0:
            wav = F.pad(wav, (1, 0))
            wav = wav[..., 1:] - self.preemphasis * wav[..., :-1]
        mel = self.melspec(wav)
        mel = self._amp_to_db(mel)
        mel = self._normalize(mel)
        return mel.to(device)

    def _normalize(self, s, headroom_db=15):
        return (s - self.min_level_db) / (-self.min_level_db + headroom_db)

    def _amp_to_db(self, x):
        return 20 * torch.log10(torch.clamp(x, min=self.hp.stft_magnitude_min))


class PreactResBlock(nn.Sequential):
    def __init__(self, dim: int):
        super().__init__(
            nn.GroupNorm(dim // 16, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(dim // 16, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + super().forward(x)


class UNetBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None, scale_factor: float = 1.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.pre_conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.res_block1 = PreactResBlock(output_dim)
        self.res_block2 = PreactResBlock(output_dim)
        self.downsample = self.upsample = nn.Identity()
        if scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor)
        elif scale_factor < 1:
            self.downsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x, h=None):
        x = self.upsample(x)
        if h is not None:
            x = x + h
        x = self.pre_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.downsample(x), x


class UNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16, num_blocks: int = 4, num_middle_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(hidden_dim * 2**i, hidden_dim * 2 ** (i + 1), scale_factor=0.5)
            for i in range(num_blocks)
        ])
        self.middle_blocks = nn.ModuleList([
            UNetBlock(hidden_dim * 2**num_blocks) for _ in range(num_middle_blocks)
        ])
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(hidden_dim * 2 ** (i + 1), hidden_dim * 2**i, scale_factor=2)
            for i in reversed(range(num_blocks))
        ])
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )

    @property
    def scale_factor(self):
        return 2 ** len(self.encoder_blocks)

    def pad_to_fit(self, x):
        hpad = (self.scale_factor - x.shape[2] % self.scale_factor) % self.scale_factor
        wpad = (self.scale_factor - x.shape[3] % self.scale_factor) % self.scale_factor
        return F.pad(x, (0, wpad, 0, hpad))

    def forward(self, x):
        shape = x.shape
        x = self.pad_to_fit(x)
        x = self.input_proj(x)

        skips = []
        for block in self.encoder_blocks:
            x, s = block(x)
            skips.append(s)

        for block in self.middle_blocks:
            x, _ = block(x)

        for block, s in zip(self.decoder_blocks, reversed(skips)):
            x, _ = block(x, s)

        x = self.head(x)
        x = x[..., : shape[2], : shape[3]]
        return x


def _normalize(x: Tensor) -> Tensor:
    return x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)


class Denoiser(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.net = UNet(3, 3)
        self.mel_fn = MelSpectrogram(hp)
        self.register_buffer("dummy", torch.zeros(1), persistent=False)

    @property
    def stft_cfg(self) -> dict:
        hop = self.hp.hop_size
        return dict(hop_length=hop, n_fft=hop * 4, win_length=hop * 4)

    def _stft(self, x: Tensor):
        dtype, device = x.dtype, x.device
        window = torch.hann_window(self.stft_cfg["win_length"], device=x.device)
        s = torch.stft(x.float(), **self.stft_cfg, window=window, return_complex=True)[..., :-1]
        mag = s.abs()
        phi = s.angle()
        cos, sin = phi.cos(), phi.sin()
        return mag.to(dtype).to(device), cos.to(dtype).to(device), sin.to(dtype).to(device)

    def _istft(self, mag: Tensor, cos: Tensor, sin: Tensor):
        real = mag * cos
        imag = mag * sin
        s = torch.complex(real, imag)
        s = F.pad(s, (0, 1), "replicate")
        window = torch.hann_window(self.stft_cfg["win_length"], device=s.device)
        x = torch.istft(s, **self.stft_cfg, window=window, return_complex=False)
        return x

    def _magphase(self, real, imag):
        mag = (real.pow(2) + imag.pow(2) + 1e-7).sqrt()
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def _predict(self, mag: Tensor, cos: Tensor, sin: Tensor):
        x = torch.stack([mag, cos, sin], dim=1)
        mag_mask, real, imag = self.net(x).unbind(1)
        mag_mask = mag_mask.sigmoid()
        real = real.tanh()
        imag = imag.tanh()
        _, cos_res, sin_res = self._magphase(real, imag)
        return mag_mask, cos_res, sin_res

    def _separate(self, mag, cos, sin, mag_mask, cos_res, sin_res):
        sep_mag = F.relu(mag * mag_mask)
        sep_cos = cos * cos_res - sin * sin_res
        sep_sin = sin * cos_res + cos * sin_res
        return sep_mag, sep_cos, sep_sin

    def forward(self, x: Tensor):
        assert x.dim() == 2, f"Expected (B, T), got {x.size()}"
        x = x.to(self.dummy)
        x = _normalize(x)
        mag, cos, sin = self._stft(x)
        mag_mask, cos_res, sin_res = self._predict(mag, cos, sin)
        sep_mag, sep_cos, sep_sin = self._separate(mag, cos, sin, mag_mask, cos_res, sin_res)
        o = self._istft(sep_mag, sep_cos, sep_sin)
        npad = x.shape[-1] - o.shape[-1]
        o = F.pad(o, (0, npad))
        return o


def load_model(checkpoint: Path, device: str = "cpu") -> Denoiser:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "module" in state:
        state = state["module"]
    model = Denoiser(HParams())
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def _inference_chunk(model: Denoiser, wav: Tensor, pad: int = 441) -> Tensor:
    length = wav.shape[-1]
    abs_max = wav.abs().max().clamp(min=1e-7)

    wav = wav.to(model.dummy.device)
    wav = wav / abs_max
    wav = F.pad(wav, (0, pad))

    out = model(wav[None])[0].cpu()
    out = out[:length]
    out = out * abs_max
    return out


def _compute_corr(x: Tensor, y: Tensor) -> Tensor:
    return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y).conj()).abs()


def _compute_offset(chunk1: Tensor, chunk2: Tensor, sr: int) -> int:
    hop_length = sr // 200
    win_length = hop_length * 4
    n_fft = 2 ** (win_length - 1).bit_length()
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=80,
        f_min=0.0,
        f_max=sr // 2,
    )

    spec1 = mel_fn(chunk1).log1p()
    spec2 = mel_fn(chunk2).log1p()

    corr = _compute_corr(spec1, spec2).mean(dim=0)

    argmax = corr.argmax().item()
    if argmax > len(corr) // 2:
        argmax -= len(corr)

    return -argmax * hop_length


def _merge_chunks(chunks: list[Tensor], chunk_len: int, hop_len: int, *, sr: int, length: int) -> Tensor:
    overlap_len = chunk_len - hop_len
    signal_len = (len(chunks) - 1) * hop_len + chunk_len
    signal = torch.zeros(signal_len, device=chunks[0].device)

    fadein = torch.linspace(0, 1, overlap_len, device=chunks[0].device)
    fadein = torch.cat([fadein, torch.ones(hop_len, device=chunks[0].device)])
    fadeout = torch.linspace(1, 0, overlap_len, device=chunks[0].device)
    fadeout = torch.cat([torch.ones(hop_len, device=chunks[0].device), fadeout])

    for i, chunk in enumerate(chunks):
        start = i * hop_len
        end = start + chunk_len

        if len(chunk) < chunk_len:
            chunk = F.pad(chunk, (0, chunk_len - len(chunk)))

        if i > 0:
            pre = chunks[i - 1][-overlap_len:]
            cur = chunk[:overlap_len]
            offset = _compute_offset(pre, cur, sr)
            start -= offset
            end -= offset

        if i == 0:
            chunk = chunk * fadeout
        elif i == len(chunks) - 1:
            chunk = chunk * fadein
        else:
            chunk = chunk * fadein * fadeout

        signal[start:end] += chunk[: signal[start:end].shape[-1]]

    return signal[:length]


def denoise_audio(
    model: Denoiser,
    wav: Tensor,
    sr: int,
    *,
    chunk_seconds: float | None = None,
    overlap_seconds: float = 1.0,
) -> tuple[Tensor, int]:
    if sr != model.hp.wav_rate:
        wav = torchaudio.functional.resample(wav, sr, model.hp.wav_rate)
        sr = model.hp.wav_rate

    if not chunk_seconds:
        return _inference_chunk(model, wav), sr

    chunk_len = int(sr * chunk_seconds)
    overlap_len = int(sr * overlap_seconds)
    hop_len = chunk_len - overlap_len

    chunks = []
    for start in range(0, wav.shape[-1], hop_len):
        chunks.append(_inference_chunk(model, wav[start : start + chunk_len]))

    out = _merge_chunks(chunks, chunk_len, hop_len, sr=sr, length=wav.shape[-1])
    return out, sr


def main():
    parser = argparse.ArgumentParser(description="Run denoiser inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Input wav file")
    parser.add_argument("--output", type=Path, required=True, help="Output wav file")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk size in seconds (0 disables chunking)",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
        help="Overlap between chunks in seconds",
    )
    args = parser.parse_args()

    wav, sr = torchaudio.load(str(args.input))
    wav = wav.mean(dim=0)

    model = load_model(args.checkpoint, device=args.device)
    chunk = args.chunk_seconds if args.chunk_seconds > 0 else None
    out, sr = denoise_audio(
        model,
        wav,
        sr,
        chunk_seconds=chunk,
        overlap_seconds=args.overlap_seconds,
    )

    torchaudio.save(str(args.output), out.unsqueeze(0), sr)


if __name__ == "__main__":
    main()

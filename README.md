# Minimal Denoiser

This repository provides a tiny subset of the original **Resemble Enhance** project.
Only the denoising model is kept together with a short script to run inference.

## Usage

```bash
pip install -r requirements.txt
python minimal_denoiser.py --checkpoint /path/to/mp_rank_00_model_states.pt \
                           --input noisy.wav --output clean.wav \
                           --chunk-seconds 30 --overlap-seconds 1

The script expects a checkpoint from the official Resemble Enhance denoiser. It
loads the model and saves the denoised waveform to the specified output file.

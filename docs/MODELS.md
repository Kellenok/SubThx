# Models

subthx downloads its models from the [subthx organization](https://huggingface.co/subthx) on Hugging Face (from the model menu in the app, or by copying the repository contents into the `models` folder). Each repository has a `manifest.json` with the size and sha256 of every file, and every download is checked against it.

## Catalog

| Model | Purpose | Source license |
|---|---|---|
| Kitsune 0.05B / 0.1B / 0.3B (by Multy123) | Japanese, anime and drama | CC BY-NC 4.0 (non-commercial use) |
| Parakeet TDT 0.6B v3, redux and ultra (NVIDIA, moondream, Olicorne) | 25 European languages | the upstream cards have no license |
| Whisper JA Anime v0.3 (efwkjn, willopcbeta) | Japanese | the upstream cards have no license |
| Whisper Large v3 Turbo (OpenAI, onnx-community) | multilingual | MIT |
| TheWhisper Large v3 Turbo (TheStageAI) | multilingual | CC BY 4.0 |
| FireRedVAD (FireRedTeam) | speech detection, always needed | Apache-2.0 |

Each repository's card on Hugging Face lists its base models, quantization and files.

## Repositories

| Repository | Contents |
|---|---|
| [kitsune-onnx](https://huggingface.co/subthx/kitsune-onnx) | Kitsune 0.05B, 0.1B, 0.3B (CTC, mixed-precision GPU and int8 CPU graphs) and the mel filterbank |
| [parakeet-v3-onnx](https://huggingface.co/subthx/parakeet-v3-onnx) | Parakeet redux and ultra (encoder and decoder graphs for GPU and CPU) and the mel filterbank |
| [whisper-ja-anime-onnx](https://huggingface.co/subthx/whisper-ja-anime-onnx) | Whisper JA Anime v0.3 (encoder and decoder graphs) |
| [whisper-large-v3-turbo-onnx](https://huggingface.co/subthx/whisper-large-v3-turbo-onnx) | Whisper Large v3 Turbo |
| [thewhisper-large-v3-turbo-onnx](https://huggingface.co/subthx/thewhisper-large-v3-turbo-onnx) | TheWhisper Large v3 Turbo |
| [firered-vad-onnx](https://huggingface.co/subthx/firered-vad-onnx) | FireRedVAD (fp32 and fp16) and its CMVN file |

The app downloads the VAD with every model, because every transcription needs it.

## Licenses

Models are not in the repository. Credit the authors listed above when you redistribute a model. Kitsune is CC BY-NC 4.0: non-commercial use only.

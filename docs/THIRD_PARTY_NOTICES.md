# Third-Party Notices

VoiceTTSr's own code is licensed under GPL-3.0 (see `LICENSE`). It depends on and bundles several third-party components with their own, different license terms. This file exists so that using or redistributing VoiceTTSr doesn't accidentally violate any of them.

## Bundled binaries (`tools/`)

### FaceFXWrapper.exe
- **Source:** [Nukem9/FaceFXWrapper](https://github.com/Nukem9/FaceFXWrapper), used to generate Bethesda `.lip` lip-sync files.
- **License:** per the upstream project's own README: *"FaceFXWrapper uses code from the Creation Kit and is subject to Bethesda's license agreement. Compressed resource files are property of Bethesda Softworks LLC."* This is **not** a standard open-source license -- redistribution and use are governed by Bethesda's Creation Kit EULA, not GPL-3.0. VoiceTTSr bundles the compiled `.exe` as distributed in Nukem9's GitHub releases.

### xWMAEncode.exe
- **Source:** Microsoft, originally distributed as part of the DirectX SDK.
- **License:** Microsoft's own redistribution terms for DirectX SDK utilities. Widely redistributed alongside Bethesda modding tools by convention, but it is Microsoft's binary, not this project's.

### FonixData.cdf -- REMOVED, do not re-add
- This file is **not bundled** in this repository (as of the 2026-07 audit remediation), and should not be re-added.
- Per FaceFXWrapper's own documentation: *"FonixData.cdf is not provided with this tool and must be obtained from the G.E.C.K."* -- it's Bethesda's proprietary data, and the upstream tooling ecosystem's convention is that every user sources their own copy from a Creation Kit install they legitimately have, not from a third-party repo.
- **If you need this file:** copy it from your own Bethesda Creation Kit install (`Data/Sound/Voice/Processing/FonixData.cdf`), or from the community-maintained "SSE CreationKit Fonixdata Lip Sync Fix" on Nexus Mods if your Creation Kit install didn't include it. Place it wherever the "Fonix" path field in the Skyrim export panel points (defaults to `tools/FonixData.cdf`).
- `skyrim_utils.py`'s `SkyrimConverter.generate_lip()` will raise a clear error with these instructions if the file isn't found.

## Python / ML dependencies

### Coqui TTS (`TTS==0.22.0`) / XTTS v2
- **License:** Coqui Public Model License (CPML) for the XTTS v2 model weights specifically -- **non-commercial use only**. The `TTS` Python package's code is more permissively licensed (MPL-2.0), but the pretrained XTTS v2 checkpoint it downloads at runtime is CPML. If you plan to use VoiceTTSr's XTTS backend commercially, check the current CPML terms at [coqui.ai](https://coqui.ai) / the model card on Hugging Face -- this project does not grant any additional rights beyond what CPML allows.

### RVC technical assets (`hubert_base.pt`, `rmvpe.pt`)
- **Source:** [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) on Hugging Face, the original RVC project's own space (see `download_resources.py`). These are the content-encoder and pitch-extraction models RVC needs to function; they are not voice/persona models.

### RVC baseline voice models -- REMOVED, do not re-add without review
- Earlier versions of `download_resources.py` auto-downloaded two "baseline" RVC voice-conversion models from a third-party mirror (`Politrees/RVC_resources`), renamed on download to `female_baseline.pth` / `male_baseline.pth`. The `male_baseline.pth` file traced back to a checkpoint named `obama.pth` on the source repo -- i.e., a voice-conversion model apparently derived from a real, named public figure, shipped with no disclosure or consent record.
- This has been removed from the default download set. See `docs/VOICE_ETHICS.md` for the policy behind this and what's required if a future contributor wants to reintroduce any bundled/auto-downloaded voice model.

### Qwen3-TTS
- **Source:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / `Qwen/Qwen3-TTS-*` model weights on Hugging Face, developed by Alibaba's Qwen team.
- Check the specific model card on Hugging Face for the license terms of whichever Qwen3-TTS checkpoint variant is in use (CustomVoice / Base / VoiceDesign, 0.6B / 1.7B).

### Chatterbox TTS
- **Source:** [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox), developed by Resemble AI. Check the upstream repo for current license terms. Note that Chatterbox embeds an imperceptible watermark (Resemble AI's "Perth" watermarker) in all generated audio by design, per upstream's own documentation.

## Summary of what this means for you

- **Using VoiceTTSr non-commercially, with your own or consented reference voices:** the license situations above are informational, not blocking.
- **Using VoiceTTSr's XTTS backend commercially:** check CPML's current terms first -- this may not be permitted without a separate agreement with Coqui.
- **Redistributing a build of VoiceTTSr:** don't bundle `FonixData.cdf`; keep `FaceFXWrapper.exe`/`xWMAEncode.exe` bundling consistent with their upstream redistribution terms; don't reintroduce a default/bundled voice model without following `docs/VOICE_ETHICS.md`.

This document reflects the state of dependencies as of the 2026-07-03 audit and its remediation. If you add, upgrade, or remove a bundled third-party component, update this file in the same change.

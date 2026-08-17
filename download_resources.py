"""
VoiceTTSr baseline resource downloader.

Fetches the technical assets RVC needs to run (content encoder + pitch
extractor). It deliberately does NOT fetch any pre-made "baseline" voice
models by default — see docs/VOICE_ETHICS.md for why, and OPTIONAL_VOICES
below if you want to opt in to a specific third-party voice model yourself.

Every entry in RESOURCES is verified against a pinned SHA-256 hash after
download, so a compromised, corrupted, or MITM'd download is rejected
instead of silently used -- the hash is the actual security boundary,
independent of which branch/URL the file is fetched from.
"""

import argparse
import hashlib
import os
import sys

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Required technical assets only. No voice/persona models are downloaded
# by default.
#
# Source: lj1995/VoiceConversionWebUI, the original RVC project's own HF
# space -- this is the canonical, widely-referenced upstream for these two
# files across the RVC ecosystem.
# ---------------------------------------------------------------------------
RESOURCES = {
    "rvc_models/hubert_base.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "sha256": "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96",
    },
    "rvc_models/rmvpe.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "sha256": "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
    },
}

# ---------------------------------------------------------------------------
# OPTIONAL, OPT-IN voice models.
#
# These are NOT downloaded by default and are not required to use VoiceTTSr
# (bring your own reference audio instead — that's the normal workflow).
# ---------------------------------------------------------------------------
OPTIONAL_VOICES = {}


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, destination: str, expected_sha256: str = None, timeout: int = 30) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.exists(destination):
        if expected_sha256:
            if _sha256_of(destination) == expected_sha256:
                print(f"[SKIP] {destination} already exists and matches expected hash.")
                return
            print(f"[WARN] {destination} exists but does NOT match expected hash — re-downloading.")
        else:
            print(f"[SKIP] {destination} already exists (no hash pinned to verify against).")
            return

    print(f"[DOWN] Fetching {os.path.basename(destination)}...")
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    tmp_destination = destination + ".partial"
    try:
        with open(tmp_destination, "wb") as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 64):
                if data:
                    size = f.write(data)
                    bar.update(size)

        if expected_sha256:
            actual = _sha256_of(tmp_destination)
            if actual != expected_sha256:
                if os.path.exists(tmp_destination):
                    os.remove(tmp_destination)
                raise ValueError(
                    f"Hash mismatch for {destination}: expected {expected_sha256}, got {actual}. "
                    "File was deleted and NOT installed. This could mean the upstream file changed "
                    "or the download was tampered with -- do not bypass this check."
                )
        else:
            print(f"[WARN] No expected hash configured for {destination}; integrity was NOT verified.")

        os.replace(tmp_destination, destination)
        print(f"[OK] Saved to {destination}")
    except Exception as e:
        if os.path.exists(tmp_destination):
            try:
                os.remove(tmp_destination)
            except Exception:
                pass
        raise e


def _print_hash(rel_path: str) -> None:
    if not os.path.exists(rel_path):
        print(f"[ERROR] {rel_path} does not exist locally; download it first.")
        sys.exit(1)
    print(_sha256_of(rel_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download OPTIONAL_VOICES entries. Off by default -- see docs/VOICE_ETHICS.md.",
    )
    parser.add_argument(
        "--print-hash",
        metavar="PATH",
        help="Print the sha256 of an already-downloaded local file, to populate RESOURCES/OPTIONAL_VOICES.",
    )
    args = parser.parse_args()

    if args.print_hash:
        _print_hash(args.print_hash)
        return

    print("\n" + "=" * 50)
    print(" VoiceTTSr Baseline Resource Downloader ")
    print("=" * 50 + "\n")

    to_fetch = dict(RESOURCES)
    if args.include_optional:
        if OPTIONAL_VOICES:
            print("[NOTE] --include-optional set: fetching opt-in voice models. Review docs/VOICE_ETHICS.md.")
            to_fetch.update(OPTIONAL_VOICES)
        else:
            print("[NOTE] --include-optional set, but no optional voices are currently configured.")

    success = True
    for dest, meta in to_fetch.items():
        try:
            download_file(meta["url"], dest, meta.get("sha256"))
        except Exception as e:
            print(f"[ERROR] Failed to download {dest}: {e}")
            success = False

    if success:
        print("\n[READY] Required resources are prepared. No default voice model was installed --")
        print("        add your own reference audio in the GUI to create a voice profile.\n")
    else:
        print("\n[WARNING] Some downloads failed. Please check your connection and re-run.\n")


if __name__ == "__main__":
    main()

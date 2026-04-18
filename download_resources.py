import os
import requests
from tqdm import tqdm

# Configuration
RESOURCES = {
    "rvc_models/hubert_base.pt": "https://huggingface.co/Politrees/RVC_resources/resolve/main/hubert_base.pt",
    "rvc_models/rmvpe.pt": "https://huggingface.co/Politrees/RVC_resources/resolve/main/rmvpe.pt",
    "rvc_models/female_baseline.pth": "https://huggingface.co/Politrees/RVC_resources/resolve/main/kizuna.pth",
    "rvc_models/male_baseline.pth": "https://huggingface.co/Politrees/RVC_resources/resolve/main/obama.pth",
}

def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        print(f"[SKIP] {destination} already exists.")
        return

    print(f"[DOWN] Fetching {os.path.basename(destination)}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" VoiceTTSr Baseline Resource Downloader ")
    print("="*50 + "\n")
    
    for dest, url in RESOURCES.items():
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"[ERROR] Failed to download {dest}: {e}")
            
    print("\n[READY] Baseline models are prepared!\n")

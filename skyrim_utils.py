import os
import subprocess
import struct
from pydub import AudioSegment

class SkyrimConverter:
    """
    Utility to handle the Skyrim SE Voice Pipeline.
    Requires FaceFXWrapper.exe, xwmaencode.exe, and FonixData.cdf.
    """
    def __init__(self, facefx_path, xwma_path, fonix_path):
        self.facefx_exe = facefx_path
        self.xwma_exe = xwma_path
        self.fonix_data = fonix_path

    def preprocess_wav(self, input_path, output_path):
        """Force 44.1kHz, 16-bit, Mono PCM for Skyrim compatibility."""
        try:
            audio = AudioSegment.from_wav(input_path)
            # Skyrim Standard: 44100Hz, 16-bit, Mono
            audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
            audio.export(output_path, format="wav")
            return output_path
        except Exception as e:
            raise Exception(f"WAV Preprocessing failed: {e}")

    def generate_lip(self, wav_path, text, output_lip_path):
        """Call FaceFXWrapper to generate .lip sync data."""
        if not os.path.exists(self.facefx_exe):
            raise FileNotFoundError(f"FaceFXWrapper not found at {self.facefx_exe}")
        
        # FaceFXWrapper [Type] [Lang] [FonixDataPath] [WavPath] [LipPath] [Text]
        # We use the 'Skip Resample' mode by providing a pre-processed 16kHz or 44kHz wav
        # Note: FaceFX often prefers 16kHz internally, but the wrapper handles it.
        cmd = [
            self.facefx_exe,
            "Skyrim",
            "USEnglish",
            self.fonix_data,
            wav_path,
            output_lip_path,
            text
        ]
        
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # FaceFXWrapper can be picky about text length/format
            raise Exception(f"FaceFX failed (Code {result.returncode}): {result.stderr or result.stdout}")
        
        if not os.path.exists(output_lip_path):
            raise Exception("FaceFX completed but NO .lip file was created.")
            
        return output_lip_path

    def encode_xwm(self, wav_path, output_xwm_path):
        """Convert WAV to XWM using xwmaencode.exe."""
        if not os.path.exists(self.xwma_exe):
            raise FileNotFoundError(f"xwmaencode.exe not found at {self.xwma_exe}")

        # Basic usage: xwmaencode.exe input.wav output.xwm
        cmd = [self.xwma_exe, wav_path, output_xwm_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"xwmaencode failed: {result.stderr or result.stdout}")
            
        return output_xwm_path

    def pack_fuz(self, lip_path, xwm_path, output_fuz_path):
        """
        Pack LIP and XWM into a .fuz container.
        FUZ Header (Bethesda Standard):
        - 4 bytes: Magic 'FUZh' (0x685A5546)
        - 4 bytes: Version (1)
        - 4 bytes: Lip size (uint32)
        - [Lip Data Bytes]
        - [Xwm Data Bytes]
        """
        try:
            with open(lip_path, "rb") as f_lip:
                lip_data = f_lip.read()
            
            with open(xwm_path, "rb") as f_xwm:
                xwm_data = f_xwm.read()

            # Header: 'FUZE', version 1, length of lip data
            header = struct.pack('<4sII', b'FUZE', 1, len(lip_data))
            
            with open(output_fuz_path, "wb") as f_out:
                f_out.write(header)
                f_out.write(lip_data)
                f_out.write(xwm_data)
            
            return output_fuz_path
        except Exception as e:
            raise Exception(f"FUZ Packing failed: {e}")

    def create_skyrim_fuz(self, source_wav, text, final_fuz_path):
        """Full pipeline: WAV -> Preprocessing -> LIP + XWM -> FUZ."""
        base_dir = os.path.dirname(final_fuz_path) or "."
        base_name = os.path.splitext(os.path.basename(final_fuz_path))[0]
        
        temp_wav = os.path.join(base_dir, f"{base_name}_temp.wav")
        temp_lip = os.path.join(base_dir, f"{base_name}_temp.lip")
        temp_xwm = os.path.join(base_dir, f"{base_name}_temp.xwm")

        try:
            # 1. Preprocess WAV for the encoder/FaceFX
            self.preprocess_wav(source_wav, temp_wav)
            
            # 2. Generate LIP Sync
            self.generate_lip(temp_wav, text, temp_lip)
            
            # 3. Create XWM Audio
            self.encode_xwm(temp_wav, temp_xwm)
            
            # 4. Pack into FUZ
            self.pack_fuz(temp_lip, temp_xwm, final_fuz_path)
            
        finally:
            # Robust Cleanup
            for f in [temp_wav, temp_lip, temp_xwm]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass

        return final_fuz_path

# Quick Test Example
if __name__ == "__main__":
    # This is just a template; paths must be provided by user config later
    # converter = SkyrimConverter(
    #     facefx_path="path/to/FaceFXWrapper.exe",
    #     xwma_path="path/to/xwmaencode.exe",
    #     fonix_path="path/to/FonixData.cdf"
    # )
    # converter.create_skyrim_fuz("test.wav", "Hello world", "test.fuz")
    print("Skyrim Utils Loaded. Ready for integration.")

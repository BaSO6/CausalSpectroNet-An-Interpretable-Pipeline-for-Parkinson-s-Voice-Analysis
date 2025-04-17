
import os
import glob
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt

def remove_generated_pngs(root_dir):
    png_files = glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True)
    for file in png_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"删除 {file} 失败：{e}")

def remove_silence(audio, sr, top_db=20):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    return np.concatenate([audio[start:end] for start, end in intervals])

def generate_stft_spectrogram(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def generate_mel_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                              hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def generate_cqt_spectrogram(audio, sr, hop_length=512, bins_per_octave=12, n_bins=84):
    C = np.abs(librosa.cqt(audio, sr=sr, hop_length=hop_length, 
                           bins_per_octave=bins_per_octave, n_bins=n_bins))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    return C_db

def save_color_spectrogram(spectrogram, output_file, cmap='viridis'):
    min_val = np.min(spectrogram)
    max_val = 0
    norm_spec = (spectrogram - min_val) / (max_val - min_val)
    cmap_obj = plt.get_cmap(cmap)
    colored = cmap_obj(norm_spec)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    image = Image.fromarray(colored_rgb)
    image.save(output_file)

def segment_audio(audio, sr, segment_length=1.0):
    segment_samples = int(sr * segment_length)
    total_samples = len(audio)
    num_segments = total_samples // segment_samples
    segments = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segments.append(audio[start:end])
    return segments

def load_progress(progress_file):
    processed = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                processed.add(line.strip())
    return processed

def update_progress(progress_file, file_path):
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")

def process_all_wav_files(root_dir, standard_dir, sample_rate=22050, n_fft=2048, hop_length=512, top_db=20):
    pd_dir = os.path.join(standard_dir, 'PD')
    hc_dir = os.path.join(standard_dir, 'HC')
    os.makedirs(standard_dir, exist_ok=True)
    os.makedirs(pd_dir, exist_ok=True)
    os.makedirs(hc_dir, exist_ok=True)

    progress_file = os.path.join(standard_dir, "progress.txt")
    processed_files = load_progress(progress_file)
    wav_files = glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True)
    wav_files_to_process = [f for f in wav_files if f not in processed_files]

    for file_path in tqdm(wav_files_to_process, desc="Processing WAV files"):
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate)
        except Exception as e:
            print(f"加载 {file_path} 失败：{e}")
            continue

        audio_nonsilence = remove_silence(audio, sr, top_db=top_db)
        segments = segment_audio(audio_nonsilence, sr, segment_length=1.0)

        lower_path = file_path.lower()
        if "pd" in lower_path:
            category = 'PD'
        elif "hc" in lower_path:
            category = 'HC'
        else:
            category = 'HC'

        for idx, segment in enumerate(segments):
            base_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_seg{idx}"
            segment_wav_file = os.path.join(standard_dir, base_name + ".wav")
            try:
                sf.write(segment_wav_file, segment, sr)
            except Exception as e:
                print(f"保存音频 {segment_wav_file} 失败：{e}")

            stft_spec = generate_stft_spectrogram(segment, sr, n_fft=n_fft, hop_length=hop_length)
            mel_spec = generate_mel_spectrogram(segment, sr, n_fft=n_fft, hop_length=hop_length)
            cqt_spec = generate_cqt_spectrogram(segment, sr, hop_length=hop_length)

            target_dir = pd_dir if category == 'PD' else hc_dir
            stft_file = os.path.join(target_dir, base_name + "_stft.png")
            mel_file = os.path.join(target_dir, base_name + "_mel.png")
            cqt_file = os.path.join(target_dir, base_name + "_cqt.png")

            try:
                save_color_spectrogram(stft_spec, stft_file)
                save_color_spectrogram(mel_spec, mel_file)
                save_color_spectrogram(cqt_spec, cqt_file)
            except Exception as e:
                print(f"保存频谱图失败（{base_name}）：{e}")

        update_progress(progress_file, file_path)

if __name__ == "__main__":
    root_wav_dir = "./Voice"
    standard_dir = os.path.join(root_wav_dir, "standard")
    remove_generated_pngs(root_wav_dir)
    process_all_wav_files(root_wav_dir, standard_dir)

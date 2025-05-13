import pandas as pd
import re
import os
import subprocess
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_silence
import warnings
import matplotlib.pyplot as plt

# Suppress librosa warnings for audioread backend issues
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# --- Configuration ---
METADATA_DIR = 'data'
AUDIO_DIR = 'audioset_raw_audio'
# --- Configuration ---
PROCESSED_DATA_DIR = 'audioset_processed_data'
SPECTROGRAM_DIR = 'audioset_mel_spectrograms'   # where individual mel spectrograms will be stored
NUM_SEGMENTS = None  # set to an int to debug on a subset, None means "process all"
# Path to cleaned metadata created earlier
CLEANED_DATA_FILE = os.path.join(METADATA_DIR, 'cleaned_data.csv')
TARGET_SAMPLE_RATE = 16000  # Hz, as per Lara's paper
AUDIO_DURATION_SECONDS = 10 # seconds, as per Lara's paper

# Mel-spectrogram parameters (from Lara's paper)
N_MELS = 128
N_FFT = 400  # window size
HOP_LENGTH = 200 # hop size (overlap)

# Create directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

# --- 1. Load Metadata from cleaned_data.csv ---
def load_cleaned_metadata(cleaned_csv):
    """
    Loads pre‑filtered metadata produced by the notebook (cleaned_data.csv).

    Expected columns in cleaned_csv:
        YTID,start_seconds,end_seconds,label_list,display_name,class

    Returns a DataFrame with canonical column names:
        ytid, start_seconds, end_seconds, instrument_name, class_name
    """
    df = pd.read_csv(cleaned_csv)

    df.rename(columns={
        'YTID': 'ytid',
        'display_name': 'instrument_name',
        'class': 'class_name'
    }, inplace=True)

    df = df[['ytid', 'start_seconds', 'end_seconds',
             'instrument_name', 'class_name']].drop_duplicates()

    print(f"Loaded {len(df)} segments from {cleaned_csv}")
    return df

# --- 2. Download Audio (using yt-dlp) and Extract Segments ---
def download_and_extract_segments(segments_df, audio_dir, processed_data_dir, target_sr, audio_duration_s):
    """
    Downloads audio for YouTube IDs and extracts specified segments.
    """
    # Ensure class sub‑directories exist when saving processed clips
    print("Starting audio download and segment extraction...")
    processed_segments_metadata = []

    for index, row in segments_df.iterrows():
        ytid = row['ytid']
        start_s = row['start_seconds']
        end_s = row['end_seconds']
        instrument_name = row['instrument_name'] # Keep the specific instrument name for this segment
        class_name = row['class_name']
        safe_class_name = re.sub(r'[^\w_]', '', class_name)
        class_dir = os.path.join(processed_data_dir, safe_class_name)
        os.makedirs(class_dir, exist_ok=True)
        raw_audio_path = os.path.join(audio_dir, f"{ytid}.wav")
        safe_instrument_name = re.sub(r'[^\w_]', '', instrument_name)
        extracted_segment_filename = f"{ytid}_{int(start_s)}_{int(end_s)}_{safe_instrument_name}.wav"
        extracted_segment_path = os.path.join(class_dir, extracted_segment_filename)

        # Skip if the processed segment already exists
        if os.path.exists(extracted_segment_path):
            print(f"Skipping {extracted_segment_filename}: Already processed.")
            processed_segments_metadata.append({
                'ytid': ytid,
                'start_seconds': start_s,
                'end_seconds': end_s,
                'instrument_name': instrument_name,
                'class_name': class_name,
                'extracted_audio_path': extracted_segment_path
            })
            continue

        # 2a. Download raw audio using yt-dlp
        if not os.path.exists(raw_audio_path):
            try:
                print(f"Downloading audio for {ytid}...")
                print(f"Attempting download URL: https://www.youtube.com/watch?v={ytid}")
                # -x: extract audio, --audio-format wav: specify format
                # -o: output filename template
                # --restrict-filenames: keep filenames simple
                subprocess.run(
                    ['yt-dlp', '-x', '--audio-format', 'wav', '--restrict-filenames',
                     '-o', raw_audio_path, f'https://www.youtube.com/watch?v={ytid}'],
                    check=True, capture_output=True, text=True
                )
                print(f"Downloaded {ytid}.wav")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {ytid}: {e.stderr}")
                continue
            except FileNotFoundError:
                print("Error: yt-dlp not found. Please install it (e.g., pip install yt-dlp) and ensure it's in your PATH.")
                return [], False # Indicate failure to continue
        else:
            print(f"Raw audio for {ytid}.wav already exists. Skipping download.")

        # 2b. Extract the specific segment and apply Lara's initial preprocessing
        try:
            audio = AudioSegment.from_file(raw_audio_path)

            # Lara's Preprocessing Step 1: Downmix to 1 channel (mono)
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Lara's Preprocessing Step 2: Resample to target_sr (pydub works on sample rate, librosa later for precise resampling)
            # Pydub's set_frame_rate resamples.
            audio = audio.set_frame_rate(target_sr)
            
            # Calculate segment start and end in milliseconds
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)

            # Extract segment
            segment = audio[start_ms:end_ms]

            # Lara's Preprocessing Step 3: Pad shorter samples to 10 seconds
            target_length_ms = audio_duration_s * 1000
            if len(segment) < target_length_ms:
                padding_needed = target_length_ms - len(segment)
                segment = segment + AudioSegment.silent(duration=padding_needed, frame_rate=target_sr)
            
            # Export the segment
            segment.export(extracted_segment_path, format="wav")
            print(f"Extracted and preprocessed segment for {ytid} ({start_s}-{end_s}s) saved to {extracted_segment_path}")

            processed_segments_metadata.append({
                'ytid': ytid,
                'start_seconds': start_s,
                'end_seconds': end_s,
                'instrument_name': instrument_name,
                'class_name': class_name,
                'extracted_audio_path': extracted_segment_path
            })

        except Exception as e:
            print(f"Error processing audio for {ytid} (segment {start_s}-{end_s}s): {e}")
            continue
    return processed_segments_metadata, True

# --- 3. Lara's Further Preprocessing & Mel-Spectrogram Generation ---
def process_and_generate_features(processed_segments_metadata, processed_data_dir, target_sr, n_mels, n_fft, hop_length, is_training_data=True):
    """
    Applies Lara's remaining preprocessing (normalization, standardization) and
    generates mel-spectrograms for the extracted audio segments.
    """
    print("Starting feature generation...")
    features_and_labels = []

    for i, meta in enumerate(processed_segments_metadata):
        audio_path = meta['extracted_audio_path']
        instrument_name = meta['instrument_name']

        try:
            # Load audio using librosa for more precise resampling and further processing
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True) # Ensure mono and target_sr

            # Lara's Preprocessing Step 4: Normalization and Standardization
            # Librosa's load already normalizes to -1 to 1 for floating point types.
            # For standardization (zero-mean, unit-variance), we can apply it after loading.
            # This step is often done per batch in deep learning frameworks,
            # but can be done here if pre-calculated statistics are used.
            # For simplicity, we'll just ensure float and within range [-1, 1] from librosa.load.
            # If a dataset-wide standardization is needed, calculate mean/std of all 'y' values.

            # Data Augmentation (for training data) - add random white noise
            if is_training_data:
                noise_amplitude = 0.005 * np.random.uniform() # Adjust noise level as needed
                y = y + noise_amplitude * np.random.normal(size=y.shape)
                
                # Clip values to prevent distortion from noise potentially exceeding -1 to 1
                y = np.clip(y, -1.0, 1.0)

            # Lara's Preprocessing Step 5: Generate Mel-Spectrograms
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # --- Save individual spectrogram ---
            spectro_class_dir = os.path.join(SPECTROGRAM_DIR, re.sub(r'[^\w_]', '', meta['class_name']))
            os.makedirs(spectro_class_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(meta['extracted_audio_path']))[0]
            np.save(os.path.join(spectro_class_dir, f"{base_name}.npy"), S_dB)

            features_and_labels.append({
                'mel_spectrogram': S_dB,
                'label': instrument_name,
                'class_name': meta['class_name'],
                'ytid': meta['ytid'],
                'start_seconds': meta['start_seconds'],
                'end_seconds': meta['end_seconds']
            })

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(processed_segments_metadata)} segments.")

        except Exception as e:
            print(f"Error generating features for {audio_path}: {e}")
            continue
    return features_and_labels

# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Load cleaned metadata
    cleaned_segments_df = load_cleaned_metadata(CLEANED_DATA_FILE)

    # Limit to a smaller number of segments for demonstration/testing due to download time
    if NUM_SEGMENTS:
        cleaned_segments_df_subset = cleaned_segments_df.head(NUM_SEGMENTS)
        print(f"Proceeding with the first {NUM_SEGMENTS} / {len(cleaned_segments_df)} segments (debug mode).")
    else:
        cleaned_segments_df_subset = cleaned_segments_df
        print(f"Proceeding with ALL {len(cleaned_segments_df_subset)} segments.")

    # Step 2: Download Audio and Extract Segments (Initial Preprocessing)
    downloaded_segments_metadata, success = download_and_extract_segments(
        cleaned_segments_df_subset, AUDIO_DIR, PROCESSED_DATA_DIR, TARGET_SAMPLE_RATE, AUDIO_DURATION_SECONDS
    )

    if success and downloaded_segments_metadata:
        # Step 3: Lara's Further Preprocessing & Mel-Spectrogram Generation
        final_processed_data = process_and_generate_features(
            downloaded_segments_metadata, PROCESSED_DATA_DIR, TARGET_SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, is_training_data=True
        )

        # --- EDA/Visualization (Example) ---
        print("\n--- Exploratory Data Analysis (EDA) of Processed Features ---")
        if final_processed_data:
            # Example: Plot the first mel-spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(final_processed_data[0]['mel_spectrogram'],
                                     sr=TARGET_SAMPLE_RATE, hop_length=HOP_LENGTH,
                                     x_axis='time', y_axis='mel', cmap='magma')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Mel-spectrogram for {final_processed_data[0]['label']} (YTID: {final_processed_data[0]['ytid']})")
            plt.tight_layout()
            plt.show()

            # Example: Check the shape of the mel-spectrograms
            example_shape = final_processed_data[0]['mel_spectrogram'].shape
            print(f"Example Mel-Spectrogram shape: {example_shape} (n_mels, frames)")

            # Example: Distribution of the processed labels (if multiple labels per segment are handled, consider unique segment labels)
            processed_labels = [item['label'] for item in final_processed_data]
            label_counts = pd.Series(processed_labels).value_counts()
            print("\nDistribution of Processed Instrument Labels:")
            print(label_counts.head(10)) # Print top 10 most frequent labels

            # --- Saving Processed Data (Example) ---
            # You would typically save these for your deep learning model
            # For example, save as numpy arrays
            mel_spectrograms = np.array([item['mel_spectrogram'] for item in final_processed_data])
            labels = np.array([item['label'] for item in final_processed_data])

            np.save(os.path.join(PROCESSED_DATA_DIR, 'mel_spectrograms.npy'), mel_spectrograms)
            np.save(os.path.join(PROCESSED_DATA_DIR, 'labels.npy'), labels)
            print(f"\nSaved {len(mel_spectrograms)} mel-spectrograms and labels to {PROCESSED_DATA_DIR}")

            # Optionally, save a mapping of label names to numerical IDs if your model needs it
            unique_labels = np.unique(labels)
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            # Save as .npy or .json as needed; json import removed, so .npy only
            np.save(os.path.join(PROCESSED_DATA_DIR, 'label_to_id.npy'), label_to_id)
            print("Saved label_to_id mapping.")

        else:
            print("No data was successfully processed for feature generation.")
    else:
        print("Audio download and segment extraction failed or yielded no segments.")
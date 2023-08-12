import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import csv

def generate_repeated_sound_csv(input_wav, output_csv, threshold=0.5):
    # Load the WAV audio file
    sample_rate, audio_signal = wavfile.read(input_wav)

    # Perform Fourier Transform to obtain frequency spectrum
    frequencies, amplitudes = signal.welch(audio_signal, fs=sample_rate)

    # Find the dominant frequency component
    dominant_index = np.argmax(amplitudes)
    dominant_frequency = frequencies[dominant_index]

    # Find repeating time samples based on the threshold
    time_samples = []
    for i in range(1, len(amplitudes)):
        if amplitudes[i] >= threshold * amplitudes[dominant_index]:
            time_samples.append(i / sample_rate)

    # Write time samples and dominant frequency to CSV
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['TimeSample', 'Frequency'])
        for time_sample in time_samples:
            csv_writer.writerow([time_sample, dominant_frequency])

# Define file paths and parameters
input_wav_file = 'gastrouble.wav'  # Convert your recorded MP3 to WAV format
output_csv_file = 'repeated_sound.csv'

# Call the function to generate the CSV file
generate_repeated_sound_csv(input_wav_file, output_csv_file)


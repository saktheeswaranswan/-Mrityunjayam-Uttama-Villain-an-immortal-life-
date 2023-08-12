import numpy as np
import scipy.signal as signal
import sounddevice as sd
import csv

def apply_notch_filter_live(input_device, output_device, csv_file, sample_rate=44100):
    # Read the CSV file with time samples and frequencies
    time_samples = []
    frequencies = []
    with open(csv_file, mode='r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            time_samples.append(float(row[0]))
            frequencies.append(float(row[1]))

    def audio_callback(indata, frames, time, status):
        nonlocal filtered_audio

        # Get the current time in seconds
        current_time = time.outputBufferDacTime

        # Find the closest time sample in the CSV
        closest_index = np.argmin(np.abs(time_samples - current_time))
        filter_frequency = frequencies[closest_index]

        # Apply the notch filter
        b, a = signal.iirnotch(filter_frequency / nyquist, Q=q_value)
        filtered_audio = signal.lfilter(b, a, indata)

        # Play the filtered audio
        outdata[:] = filtered_audio[:, np.newaxis]

    # Initialize audio I/O
    nyquist = 0.5 * sample_rate
    q_value = 10
    filtered_audio = np.zeros((1024,))  # Initialize with zeros
    with sd.Stream(device=(input_device, output_device), callback=audio_callback, channels=1):
        sd.sleep(100000)  # Run the stream for a longer duration

# Define audio devices and parameters
input_device = None  # Use the default input device
output_device = None  # Use the default output device
csv_file = 'repeated_sound.csv'  # CSV file with time samples and frequencies

# Call the function to apply the notch filter to live microphone input
apply_notch_filter_live(input_device, output_device, csv_file)


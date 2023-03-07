import os
import librosa
import soundfile as sf

def load(path, sr):
    return librosa.load(path, sr=sr)[0]
    
def write_wav(path, signal, sr):
    sf.write(path, signal, sr, subtype="PCM_24")

def estimate_beats(signal, sr):
    beats =  librosa.beat.beat_track(signal, sr, units="samples")[1]
    return beats
    
def cut(signal, delimiters):
    segments = []
    start_sample = 0
    for delimiter in delimiters:
        stop_sample = delimiter
        segment = signal[start_sample:stop_sample]
        segments.append(segment)
        start_sample = stop_sample
    last_segment = signal[start_sample:]
    segments.append(last_segment)
    return segments

class SegmentExtractor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self._audio_format = "wav"

    def create_and_save_segments(self, dir, save_dir):
        for root, _, files in os.walk(dir):
            for file in files:
                self._create_and_save_segments_for_file(file, dir, save_dir)

    def _create_and_save_segments_for_file(self, file, root, save_dir):
        file_path = os.path.join(root, file)
        signal = load(file_path, self.sample_rate)
        beat_events = estimate_beats(signal, self.sample_rate)
        segments = cut(signal, beat_events)
        self._write_segments_to_wav(file, save_dir, segments)
        print(f"Beats saved for {file_path}")

    def _write_segments_to_wav(self, file, save_dir, segments):
        for i, segment in enumerate(segments):
            save_path = self._generate_save_path(file, save_dir, i)
            write_wav(save_path, segment, self.sample_rate)

    def _generate_save_path(self, file, save_dir, num_segment):
        file_name = f"{file}_{num_segment}.{self._audio_format}"
        save_path = os.path.join(save_dir, file_name)
        return save_path

def segment(load_dir, save_dir, sample_rate):
    segment_extractor = SegmentExtractor(sample_rate)
    segment_extractor.create_and_save_segments(load_dir, save_dir)

Load_dir = "/home/mukesh/Desktop/infiniteremixer/songs/audio"
Save_dir = "/home/mukesh/Desktop/infiniteremixer/songs/gen_audio"
segment(Load_dir, Save_dir, 22050)
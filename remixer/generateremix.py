import pickle
from collections.abc import Sequence
import soundfile as sf
import librosa
import numpy as np, pickle
import random, math, os
from dataclasses import dataclass
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from scipy.spatial import distance


class Siamese:
    def __init__(self):
        self.model = None
        self.frames_train = None
        self.mapping = None

    def get_closest(self, sample, num_neighbours=1):
        encoded = self.model.predict(np.array(self.frames_train[0]).transpose(0,2,1))
        sim = self.model.predict(np.array([sample]).transpose(0,2,1))

        out = [distance.euclidean(sim[0], encoded[x]) for x in range(len(encoded))]
        array_indexes = np.argsort(out)[::-1]
        paths = self._from_indexes_to_paths(array_indexes[:num_neighbours])
        return paths, None

    def _from_indexes_to_paths(self, sample_indexes):
        paths = [self.mapping[x] for x in sample_indexes]
        return paths

class NNSearch:
    def __init__(self):
        self.model = None
        self.mapping = None

    def get_closest(self, sample, num_neighbours=1):
        sample = sample[np.newaxis, ...]
        distances, array_indexes = self.model.kneighbors(sample,
                                                         n_neighbors=num_neighbours)
        paths = self._from_indexes_to_paths(array_indexes[0])
        return paths, distances[0]

    def _from_indexes_to_paths(self, sample_indexes):
        paths = [self.mapping[x] for x in sample_indexes]
        return paths

def load_from_pickle(load_path):
    with open(load_path, "rb") as file:
        data = pickle.load(file)
    return data


def write_wav(path, signal, sr):
    sf.write(path, signal, sr, subtype="PCM_24")


def concatenate_arrays(arrays):
    return np.hstack(arrays)

class AudioChunkMerger:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def concatenate(self, audio_file_paths):
        time_series = [self.load_audio_file(file) for file in audio_file_paths]
        concatenated_time_series = concatenate_arrays(time_series)
        return concatenated_time_series

    def load_audio_file(self, audio_file_path):
        return librosa.load(audio_file_path, sr=self.sample_rate)[0]

class FeatureRetriever:
    def __init__(self):
        self.mapping = None
        self.features = None

    def get_feature_vector(self, file_path):
        array_index = self._from_path_to_index(file_path)
        feature_vector = self.features[array_index]
        return feature_vector

    def _from_path_to_index(self, file_path):
        return self.mapping.index(file_path)

class Remix(Sequence):
    def __init__(self, *beats):
        self._beats = list(beats)

    def __len__(self):
        return len(self._beats)

    def __getitem__(self, item):
        return self._beats.__getitem__(item)

    @property
    def last_beat(self):
        num_beats_in_remix = len(self._beats)
        return self._beats[num_beats_in_remix-1]

    @property
    def num_beats_with_last_track(self):
        if len(self._beats) == 1:
            return 1
        num_beats_with_last_track = 0
        previous_beat_track = self.last_beat.track
        for i, beat in enumerate(reversed(self._beats)):
            num_beats_with_last_track += 1
            if beat.track != previous_beat_track:
                return num_beats_with_last_track
            previous_beat_track = beat.track
            if i == len(self) - 1:
                return num_beats_with_last_track

    @property
    def file_paths(self):
        file_paths = [beat.file_path for beat in self._beats]
        return file_paths

    def append(self, beat):
        self._beats.append(beat)

    def reset(self):
        self._beats = []

class Remixer:
    def __init__(self, number_of_beats):
        self.number_of_beats = number_of_beats
        self.beat_selector = None

    def generate_remix(self):
        remix = Remix()
        for _ in range(self.number_of_beats):
            beat = self._choose_beat(remix)
            remix.append(beat)
        return remix

    def _choose_beat(self, remix):
        return self.beat_selector.choose_beat(remix)

@dataclass
class Beat:
    file_path: str
    track: str
    number: int

    @classmethod
    def from_file_path(cls, file_path):
        track = Beat._get_track_from_file_path(file_path)
        beat_number = Beat._get_beat_number_from_file_path(file_path)
        return Beat(file_path, track, beat_number)

    @staticmethod
    def replace_number_in_file_path(file_path, number):
        number_and_format = file_path.split("_")[-1]
        format = file_path.split(".")[-1]
        new_number_and_format = f"{number}.{format}"
        path_head = file_path[:-len(number_and_format)]
        new_file_path = path_head + new_number_and_format
        return new_file_path

    @staticmethod
    def _get_beat_number_from_file_path(file_path):
        file_name = os.path.split(file_path)[1]
        number_and_format = file_name.split("_")[-1]
        number = number_and_format.split(".")[0]
        return int(number)

    @staticmethod
    def _get_track_from_file_path(file_path):
        file_name = os.path.split(file_path)[1]
        number_and_format = file_name.split("_")[-1]
        num_characters_to_drop = len(number_and_format) + 1
        track = file_name[:-num_characters_to_drop]
        return track

class BeatSelector:
    def __init__(self, jump_rate):
        self.jump_rate = jump_rate
        self.nn_search = None
        self.feature_retriever = None
        self.beat_file_paths = None

    def choose_beat(self, remix):
        if len(remix) == 0:
            return self._choose_first_beat()
        if self._is_beat_jump(remix.num_beats_with_last_track):
            if MODEL == "knn":
                return self._choose_beat_with_jump_knn(remix.last_beat)
            else:
                return self._choose_beat_with_jump_snn(remix.last_beat)
        return self._get_next_beat_in_track_if_possible_or_random(remix.last_beat)

    def _choose_first_beat(self):
        return self._choose_beat_randomly()

    def _choose_beat_randomly(self):
        chosen_beat_file_path = random.choice(self.beat_file_paths)
        return Beat.from_file_path(chosen_beat_file_path)

    def _is_beat_jump(self, num_beats_with_last_track):
        threshold = self._calculate_jump_threshold(num_beats_with_last_track)
        if random.random() <= threshold:
            return True
        return False

    def _calculate_jump_threshold(self, num_beats_with_last_track):
        if num_beats_with_last_track > 0:
            return self.jump_rate * math.log(num_beats_with_last_track, 10)
        return 0

    def _choose_beat_with_jump_snn(self, last_beat):
        y, sr = librosa.load(last_beat.file_path)
        frame_len = SAMPLE_RATE

        # Normalize time series
        y = ((y-np.amin(y))*2)/(np.amax(y) - np.amin(y)) - 1
        # Remove silence from the audio
        intervals = librosa.effects.split(y, top_db= 15, ref= np.max)
        intervals = intervals.tolist()
        y = (y.flatten()).tolist()
        nonsilent_y = []

        for p,q in intervals :
            nonsilent_y = nonsilent_y + y[p:q+1] 

        y = np.array(nonsilent_y)

        for j in range(0, len(y), int(frame_len)) :
            frame = y[j:j+frame_len]
            if len(frame) < frame_len :
                frame = frame.tolist() + [0]* (frame_len-len(frame))
            frame = np.array(frame)
            S = np.abs(librosa.stft(frame, n_fft=512))

        next_beat_file_paths, _ = self.nn_search.get_closest(S, 500)
        next_beat = self._get_closest_beat_of_different_track(next_beat_file_paths, last_beat)
        return next_beat

    def _choose_beat_with_jump_knn(self, last_beat):
        feature_vector = self.feature_retriever.get_feature_vector(last_beat.file_path)
        next_beat_file_paths, _ = self.nn_search.get_closest(feature_vector, 500)
        next_beat = self._get_closest_beat_of_different_track(next_beat_file_paths, last_beat)
        return next_beat

    def _get_closest_beat_of_different_track(self, beat_file_paths, last_beat):
        for file_path in beat_file_paths:
            beat = Beat.from_file_path(file_path)
            if beat.track != last_beat.track:
                return beat
        return Beat.from_file_path(beat_file_paths[0])

    def _get_next_beat_in_track_if_possible_or_random(self, beat):
        next_number = beat.number + 1
        next_beat_file_path = Beat.replace_number_in_file_path(beat.file_path,
                                                               next_number)
        if next_beat_file_path in self.beat_file_paths:
            return Beat.from_file_path(next_beat_file_path)
        return self._choose_beat_randomly()

def generate_remix(jump_rate, number_of_beats, save_path):
    jump_rate = float(jump_rate)
    num_of_beats = int(number_of_beats)

    remixer, chunk_merger = _create_objects(jump_rate, num_of_beats)
    remix = remixer.generate_remix()
    print(f"Generated remix with {num_of_beats} beats")
    audio_remix = chunk_merger.concatenate(remix.file_paths)
    print(f"Merged beats together")
    write_wav(save_path, audio_remix, SAMPLE_RATE)
    print(f"Saved new remix to {save_path}")


def _create_objects(jump_rate, number_of_beats):
    features = load_from_pickle(FEATURES_PATH)

    chunk_merger = AudioChunkMerger()
    feature_retriever = FeatureRetriever()
    feature_retriever.features = features

    if(MODEL=="knn"):
        nn_search = NNSearch()
        nearest_neighbour_model = load_from_pickle(NEAREST_NEIGHBOUR_PATH)
        beats_file_paths = load_from_pickle(MAPPING_PATH_KNN)
        nn_search.mapping = beats_file_paths
        nn_search.model = nearest_neighbour_model
    else:
        nn_search = Siamese()
        beats_file_paths = load_from_pickle(MAPPING_PATH_SNN)
        nn_search.mapping = beats_file_paths
        nn_search.model = load_model("/home/mukesh/Desktop/infiniteremixer/encoder.h5", compile= False)
        nn_search.frames_train = load_from_pickle("/home/mukesh/Desktop/infiniteremixer/training_frames.pkl")
    beat_selector = BeatSelector(jump_rate)
    beat_selector.nn_search = nn_search
    beat_selector.feature_retriever = feature_retriever
    feature_retriever.mapping = beats_file_paths
    beat_selector.beat_file_paths = beats_file_paths
    remixer = Remixer(number_of_beats)
    remixer.beat_selector = beat_selector

    return remixer, chunk_merger

MAPPING_PATH_KNN = "/home/mukesh/Desktop/infiniteremixer/songs/gen_data/mapping.pkl"
MAPPING_PATH_SNN = "/home/mukesh/Desktop/infiniteremixer/songs/gen_data/mapping_snn.pkl"
FEATURES_PATH = "/home/mukesh/Desktop/infiniteremixer/songs/gen_data/dataset.pkl"
NEAREST_NEIGHBOUR_PATH = "/home/mukesh/Desktop/infiniteremixer/songs/gen_data/nearestneighbour.pkl"
MODEL = "snn"
SAMPLE_RATE = 22050

if __name__ == "__main__":
    jump_rate = 0.01
    number_of_beats = 100
    save_path = "/home/mukesh/Desktop/infiniteremixer/songs/gen_remix/remix.wav"
    generate_remix(jump_rate, number_of_beats, save_path)
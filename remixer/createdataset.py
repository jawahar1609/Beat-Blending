import argparse, librosa
import numpy as np
from abc import ABC, abstractmethod

def load(path, sr):
    return librosa.load(path, sr=sr)[0]


class BatchExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.extractors = []
        self._features = {}

    def add_extractor(self, extractor):
        self.extractors.append(extractor)

    def extract(self, dir):
        features = {}
        for root, _, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_features = self._extract_features_for_file(file_path)
                features[file_path] = file_features
        return features

    def _extract_features_for_file(self, file_path):
        features = {}
        signal = load(file_path, self.sample_rate)
        for extractor in self.extractors:
            feature = extractor.extract(signal, self.sample_rate)
            features[extractor.feature_name] = feature
        return features

class Extractor(ABC):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    @abstractmethod
    def extract(self, signal, sample_rate):
        pass

class ChromogramExtractor(Extractor):
    def __init__(self, frame_size=1024, hop_length=512):
        super().__init__("chromogram")
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal, sample_rate):
        chromogram = librosa.feature.chroma_stft(signal,
                                                 n_fft=self.frame_size,
                                                 hop_length=self.hop_length,
                                                 sr=sample_rate)
        return chromogram

class MFCCExtractor(Extractor):
    def __init__(self, frame_size=1024, hop_length=512, num_coefficients=13):
        super().__init__("mfcc")
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.num_coefficients = num_coefficients

    def extract(self, signal, sample_rate):
        mfcc = librosa.feature.mfcc(signal,
                                    n_mfcc=self.num_coefficients,
                                    n_fft=self.frame_size,
                                    hop_length=self.hop_length,
                                    sr=sample_rate)
        return mfcc

class BatchAggregator(ABC):
    def __init__(self):
        self.aggregators = []

    def add_aggregator(self, aggregator):
        self.aggregators.append(aggregator)

    @abstractmethod
    def aggregate(self, array):
        pass

class FlatBatchAggregator(BatchAggregator):
    def aggregate(self, array):
        merged_aggregations = []
        for aggregator in self.aggregators:
            aggregation = aggregator.aggregate(array)
            merged_aggregations.append(aggregation)
        return concatenate_arrays(merged_aggregations)

class Aggregator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def aggregate(self, array):
        pass


class MeanAggregator(Aggregator):
    def __init__(self, aggregation_axis):
        super().__init__("mean")
        self.aggregation_axis = aggregation_axis

    def aggregate(self, array):
        return np.mean(array, axis=self.aggregation_axis)

class MultiTrackBatchAggregator:
    def __init__(self):
        self.batch_aggregator = None

    def aggregate(self, tracks_features):
        tracks_aggregations = {}
        for track_path, track_features in tracks_features.items():
            features_aggregations = {}
            for feature_type, features in track_features.items():
                aggregations = self.batch_aggregator.aggregate(features)
                features_aggregations[feature_type] = aggregations
            tracks_aggregations[track_path] = features_aggregations
        return tracks_aggregations


def concatenate_arrays(arrays):
    return np.hstack(arrays)

class FeatureMerger:
    def merge(self, tracks_features):
        merged_features = {}
        for track_path, track_features in tracks_features.items():
            track_merged_features = self._merge_features_for_track(track_features)
            merged_features[track_path] = track_merged_features
        return merged_features

    def _merge_features_for_track(self, track_features):
        features = [feature for feature in track_features.values()]
        merged_features = concatenate_arrays(features)
        return merged_features

class DataPreparer:
    def __init__(self):
        pass

    def prepare_mapping_and_dataset(self, features):
        mapping = self._prepare_mapping(features)
        dataset = self._prepare_dataset(features)
        return mapping, dataset

    def _prepare_mapping(self, features):
        return list(features.keys())

    def _prepare_dataset(self, features):
        dataset = list(features.values())
        dataset = np.asarray(dataset)
        return dataset

import os, pickle

def save_to_pickle(save_path, data):
    with open(save_path, "wb") as file:
        pickle.dump(data, file)

class DatasetPipeline:
    def __init__(self):
        self.batch_extractor = None
        self.multi_track_batch_aggregator = None
        self.feature_merger = None
        self.data_preparer = None

    def process(self, dir, save_dir):
        tracks_features = self.batch_extractor.extract(dir)
        print("Extracted features")
        tracks_aggregations = self.multi_track_batch_aggregator.aggregate(tracks_features)
        print("Performed statistical aggregation of features")
        tracks_merged_features = self.feature_merger.merge(tracks_aggregations)
        print("Merged multiple features")
        mapping, dataset = self.data_preparer.prepare_mapping_and_dataset(tracks_merged_features)
        print("Prepared mapping and dataset")
        mapping_path = self._save_data(save_dir, mapping, "mapping")
        print(f"Saved mapping to {mapping_path}")
        dataset_path = self._save_data(save_dir, dataset, "dataset")
        print(f"Saved dataset to {dataset_path}")

    def _save_data(self, save_dir, data, data_type):
        save_path = os.path.join(save_dir, f"{data_type}.pkl")
        save_to_pickle(save_path, data)
        return save_path

def _create_data_pipeline():
    batch_extractor = BatchExtractor()
    chromogram_extractor = ChromogramExtractor()
    batch_extractor.add_extractor(chromogram_extractor)
    mfcc_extractor = MFCCExtractor()
    batch_extractor.add_extractor(mfcc_extractor)

    batch_aggregator = FlatBatchAggregator()
    mean_aggregator = MeanAggregator(1)
    batch_aggregator.add_aggregator(mean_aggregator)

    mtba = MultiTrackBatchAggregator()
    mtba.batch_aggregator = batch_aggregator

    feature_merger = FeatureMerger()
    data_preparer = DataPreparer()

    dataset_pipeline = DatasetPipeline()
    dataset_pipeline.batch_extractor = batch_extractor
    dataset_pipeline.multi_track_batch_aggregator = mtba
    dataset_pipeline.feature_merger = feature_merger
    dataset_pipeline.data_preparer = data_preparer

    return dataset_pipeline

def create_dataset(load_dir, save_dir):
    data_pipeline = _create_data_pipeline()
    data_pipeline.process(load_dir, save_dir)

Load_dir = "/home/mukesh/Desktop/infiniteremixer/songs/gen_audio"
Save_dir = "/home/mukesh/Desktop/infiniteremixer/songs/gen_data"
create_dataset(Load_dir, Save_dir)
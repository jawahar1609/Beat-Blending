import argparse
import pickle

from sklearn.neighbors import NearestNeighbors

def save_to_pickle(save_path, data):
    with open(save_path, "wb") as file:
        pickle.dump(data, file)


def load_from_pickle(load_path):
    with open(load_path, "rb") as file:
        data = pickle.load(file)
    return data

def fit_nearest_neighbours(dataset_path, save_path):
    dataset = load_from_pickle(dataset_path)
    print(f"Dataset array has shape {dataset.shape}")
    nearest_neighbour = NearestNeighbors()
    nearest_neighbour.fit(dataset)
    print("Created nearest neighbour")
    save_to_pickle(save_path, nearest_neighbour)

Dataset_path = "/home/mukesh/Desktop/infiniteremixer/songs/gen_data/dataset.pkl"
Save_path = "/home/mukesh/Desktop/infiniteremixer/songs/gen_data/nearestneighbour.pkl"

fit_nearest_neighbours(Dataset_path, Save_path)
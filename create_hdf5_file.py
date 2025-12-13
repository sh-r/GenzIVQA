import h5py
import numpy as np

def create_hdf5_file(file_path, dict_data):
    with h5py.File(file_path, "w") as hdf_file:
        
        for batch_idx, batch in dict_data.items():
            hdf_file.create_dataset(f'datas-{batch_idx}', data = batch['datas'])
            hdf_file.create_dataset(f'mos_scores-{batch_idx}', data = batch['mos_scores'])
            hdf_file.create_dataset(f'filenames-{batch_idx}', data = batch['filenames'])

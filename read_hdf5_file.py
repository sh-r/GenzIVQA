import h5py
import numpy as np
def read_hdf5_file(file_path):
    file_path = file_path
    loaded_data = {}
    with h5py.File(file_path, "r") as hdf_file:
        for key in hdf_file.keys():
            keyname, batch_idx = key.split("-")
            batch_idx = int(batch_idx)
            if batch_idx not in loaded_data:
                loaded_data[batch_idx] = {"datas": {}, "mos_scores": {}, "filenames": {}}

            loaded_data[batch_idx][keyname] = np.array(hdf_file[key])

    return loaded_data

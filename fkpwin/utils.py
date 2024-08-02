from numpy import array_equal
import h5py

def get_dict_from_hdf5(group, none_flag=float("nan")):
    d = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            d[key] = get_dict_from_hdf5(group[key], none_flag=none_flag)
        elif isinstance(group[key], h5py.Dataset):
            data = group[key][()]

            if array_equal(data, none_flag):
                d[key] = None
            else:
                d[key] = data
    return d

def save_dict_to_hdf5(group, data, none_flag=float("nan")):
    def save_recursive(subgroup, subdata):
        for key, value in subdata.items():
            if isinstance(value, dict):
                save_recursive(subgroup.create_group(key), value)
            elif value is None:
                subgroup.create_dataset(key, data=none_flag)
            else:
                subgroup.create_dataset(key, data=value)

    save_recursive(group, data)

def save(filename, dictionary):
    with h5py.File(filename + '.h5', 'w') as hf: save_dict_to_hdf5(hf, dictionary)

def read(filename):
    with h5py.File(filename + '.h5', 'r') as hf: d = get_dict_from_hdf5(hf)
    return d
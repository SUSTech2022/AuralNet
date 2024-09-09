from tqdm import tqdm
import mat73
import numpy as np
import h5py
import os

def formulate_input(data_list,lable):
    """
    formulate input data for training, given data_list and label data

    Parameters
    ----------
    data_list: list
        list of data, each data is a list of 3 numpy arrays, 
        represents the input data of the model
    lable: numpy array
        label data corresponding to data_list

    Returns
    -------
    x1, x2, x3, lable: numpy arrays
        x1, x2, x3 are the input data for the model, 
        lable is the label data
    """
    x1 = np.array([item[0] for item in data_list]).transpose(0, 2, 1)
    print(f"x1.shape: {x1.shape}")
    x2 = np.array([item[1]  for item in data_list]).transpose(0, 2, 1)
    print(f"x2.shape: {x2.shape}")
    x3 = np.array([item[2]  for item in data_list])
    print(f"x3.shape: {x3.shape}")
    print(f"lable.shape: {lable.shape}")
    return [x1, x2, x3], lable

def prepare_dataset(data_paths,label_paths,save_path):
    """
    prepare data for training

    Parameters
    ----------
    data_paths: list
        paths of input data files
    label_paths: list
        paths of label data files
    save_path: str
        path to save the prepared data

    Returns
    -------
    """
    input_data = []
    for path in tqdm(data_paths, desc="Loading input data"):
        data = mat73.loadmat(path)['features']
        input_data.extend(data)
    
    lable = []
    for path in tqdm(label_paths, desc="Loading  lable"):
        data = mat73.loadmat(path)['labels']
        lable.append(data)
    lable_concat = np.concatenate(lable,axis = 0)

    x, y = formulate_input(input_data,lable_concat)

    with h5py.File(save_path, 'w') as h5f:
        h5f.create_dataset('x1', data=x[0])
        h5f.create_dataset('x2', data=x[1])
        h5f.create_dataset('x3', data=x[2])
        h5f.create_dataset('y', data=y)
    print(f"Saved data to {save_path}")

if __name__ == '__main__':
    save_dir = "./dataset/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    train_matpaths = [
        "./1src_clean_dataset/MRT05_1Source_TRAIN_clean_features_33gcc_64gtf_50ms.mat",
        "./2src_clean_dataset/MRT05_2Source_TRAIN_clean_features_33gcc_64gtf_50ms.mat",
        "./3src_clean_dataset/MRT05_3Source_TRAIN_clean_features_33gcc_64gtf_50ms.mat"]
    train_labelpaths = [
        "./1src_clean_dataset/MRT05_1Source_TRAIN_clean_labels_tanh_new.mat",
        "./2src_clean_dataset/MRT05_2Source_TRAIN_clean_labels_tanh_new.mat",
        "./3src_clean_dataset/MRT05_3Source_TRAIN_clean_labels_tanh_new.mat"]
    
    val_matpaths = [
        "./1src_clean_dataset/MRT05_1Source_VAL_clean_features_33gcc_64gtf_50ms.mat",
        "./2src_clean_dataset/MRT05_2Source_VAL_clean_features_33gcc_64gtf_50ms.mat",
        "./3src_clean_dataset/MRT05_3Source_VAL_clean_features_33gcc_64gtf_50ms.mat"]
    val_labelpaths = [
        "./1src_clean_dataset/MRT05_1Source_VAL_clean_labels_tanh_new.mat",
        "./2src_clean_dataset/MRT05_2Source_VAL_clean_labels_tanh_new.mat",
        "./3src_clean_dataset/MRT05_3Source_VAL_clean_labels_tanh_new.mat"]
    prepare_dataset(train_matpaths,train_labelpaths,save_path=save_dir+"train_clean.h5")
    prepare_dataset(val_matpaths,val_labelpaths,save_path=save_dir+"train_clean.h5")


    MCT_train_matpaths = ["",
                          "",
                          ""]
    MCT_train_labelpaths = ["",
                            "",
                            ""]
    MCT_val_matpaths = ["",
                        "",
                        ""]
    MCT_val_labelpaths = ["",
                          "",
                          ""]
    prepare_dataset(MCT_train_matpaths,MCT_train_labelpaths,save_path=save_dir+"MCT_train.h5")
    prepare_dataset(MCT_val_matpaths,MCT_val_labelpaths,save_path=save_dir+"MCT_val.h5")



    prepare_dataset([""],[""],save_path=save_dir+"test_0db_1src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_10db_1src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_20db_1src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_anechoic_1src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_0db_2src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_10db_2src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_20db_2src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_anechoic_2src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_0db_3src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_10db_3src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_20db_3src.h5")
    prepare_dataset([""],[""],save_path=save_dir+"test_anechoic_3src.h5")





        

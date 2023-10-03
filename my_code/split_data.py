def split_data(datapath, train_data_ratio, val_data_ratio, test_data_ratio):
    # this function will split the data
    # https://youtu.be/C6wbr1jJvVs
    """
    pip install split-folders
    """

    import splitfolders  # or import split_folders

    #input_folder = 'cell_images/'

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    #Train, val, test
    splitfolders.ratio(datapath, output="/home/abidhasan/Documents/Indicate_FH/train_val_test", seed=42, ratio=(train_data_ratio, val_data_ratio, test_data_ratio), group_prefix=None) # default values


    # Split val/test with a fixed number of items e.g. 100 for each set.
    # To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
    # enable oversampling of imbalanced datasets, works only with fixed
    """
    splitfolders.fixed(datapath, output="data2", seed=42, fixed=(35, 20), oversample=False, group_prefix=None) 
    """

    
    

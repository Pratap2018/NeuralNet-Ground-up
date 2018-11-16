# import required libraries
import torch
import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader,Dataset
def create_meta_csv(dataset_path, destination_path):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """

    # Change dataset path accordingly
    DATASET_PATH = os.path.abspath(dataset_path)
    print(DATASET_PATH)
    
    

    if not os.path.exists(os.path.join(DATASET_PATH, "/dataset_attr.csv")):

        # Make a csv with full file path and labels
        labels=os.listdir(DATASET_PATH+"/")
        print(labels)
        label=[]
        value=[]
        l_idx=[]
        k=0
        for i in labels:
            if i=='dataset_attr.csv' :
                continue
            for j in os.listdir(DATASET_PATH+"/"+i+"/"):
                l_idx.append(k)
                label.append([i,k])
                value.append(DATASET_PATH+"/"+i+"/"+j)
            k=k+1
        print(np.unique(l_idx))    
                
        # change destination_path to DATASET_PATH if destination_path is None 
        if destination_path == None:
            destination_path = dataset_path

        # write out as dataset_attr.csv in destination_path directory
        df=pd.DataFrame({'path':value,'label':label,'label_idx':l_idx})
        df.to_csv(destination_path+"/"+"dataset_attr.csv")

        # if no error
        return True

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a 
    fraction in split parameter.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    if create_meta_csv(dataset_path, destination_path=destination_path):
        dframe = pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'))

    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        # shuffle the dataframe here
        dframe=dframe.sample(frac=1)
        #pass

    if split != None:
        train_set, test_set = train_test_split(dframe, split)
        return dframe, train_set, test_set 
    
    return dframe

def train_test_split(dframe, split_ratio):
    """Splits the dataframe into train and test subset dataframes.

    Args:
        split_ration (float): Divides dframe into two splits.

    Returns:
        train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
        test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
    """
    # divide into train and test dataframes
    split_train=int(split_ratio*len(dframe))
    train_data=dframe[:split_train]
    test_data=dframe[split_train:]
    return train_data, test_data


class ImageDataset(Dataset):
    """Image Dataset that works with images
    
    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.
    
    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """

    def __init__(self, data, transform=None):
        
        self.data = data
        self.transform = transform
        self.classes = data['label'].unique()# get unique classes fr4  cgjom data dataframe
        self.label_idx=data['label_idx']
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        image = Image.open(img_path)# load PIL image
        label = self.data.iloc[idx]['label']# get label (derived from self.classes; type: int/long) of image
        label_idx=self.data.iloc[idx]['label_idx']
        if self.transform:
            image = self.transform(image)
            
        return image, label,label_idx


if __name__ == "__main__":
    # test config
    dataset_path = './Data/fruits'
    dest = './Data/Dataset'
    classes = 5
    total_rows = 4323
    randomize = True
    clear = True
    
    # test_create_meta_csv()
    df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path,destination_path=dest, randomize=randomize, split=0.99)
    print(df.describe())
    print(trn_df.describe())
    print(tst_df.describe())
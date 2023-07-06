import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import os
import csv
import random
import numpy as np
import cv2

NUM_WORKERS = os.cpu_count()

def create_datasets():
    # Initialize the data sets
    # old way
    """
    full_ds = ISLESDataSet()
    split_index = int(len(full_ds) * 0.8)  
    train_ds, val_ds = torch.utils.data.random_split(full_ds, (split_index,len(full_ds)-split_index))
    """

    # new way
    pat_list = [1,2,3,5,9,10,12,13,15,16,17,19,21,22,23,24,25,26,27,28]
    random.shuffle(pat_list)
    patids = random.choices(pat_list, k=3)
    train_ds = ISLESDataSet(patids=patids) #, max_size = 256)  # delete max_size if enough memory available
    val_ds = ISLESDataSet(val=True, patids=patids) #, max_size = 32)  # delete max_size if enough memory available

    print('Training Dataset: {} images'.format(len(train_ds)))
    print('Testing Dataset: {} images'.format(len(val_ds)))
    return train_ds, val_ds

def create_dataloaders(
        train_ds,
        val_ds,
        batch_size:int,
        num_workers:int=NUM_WORKERS,
    ):
    # Creates training and testing DataLoaders.
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # avoids unnecessary copying of memory between CPU and GPU memory by "pinning" examples that have been seen before,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader

class ISLESDataSet(Dataset):
    def __init__(self, train=True, val=False, preload=False, patid=None, patids=None, max_size=None):
        super(ISLESDataSet, self).__init__()
        pp = os.getcwd()
        
        # flag for training or test set
        if train:
            self.path = pp + '/data/ISLES2015_Slices/TrainData/'
            csv_file = os.path.join(self.path, 'ISLES2015_Slices_Training.csv')
        else:
            self.path = pp + '/data/ISLES2015_Slices/TestData/'
            csv_file = os.path.join(self.path, 'ISLES2015_Slices_Testing.csv')
        # set given val for augmentation decision
        self.val = val
        # read info from CSV file into a list of dicts
        self.filenames = self.read_csv_file(csv_file)
        # if val and patids given, only keep images of all patients in patids
        if self.val and patids is not None:
            files = []
            for id in patids:
                patid_str = "Training_{:03d}".format(id)
                patid_files = [f_dict for f_dict in self.filenames if patid_str in f_dict['flair']]
                files.extend(patid_files)
            self.filenames = files
        # if patids given, delete images of all patients in patids
        elif patids is not None:
            for id in patids:
                patid_str = "Training_{:03d}".format(id)
                self.filenames = [f_dict for f_dict in self.filenames if patid_str not in f_dict['flair'] ]
                assert len(self.filenames) > 0, f"Unknown patient id {id} !"
        # if patid is given, only keep images of this patient    
        if patid is not None:
            assert patids is None, f"Parameter 'patids' must be None to retrieve only images of one patient !"
            patid_str = "Training_{:03d}".format(patid)
            self.filenames = [f_dict for f_dict in self.filenames if patid_str in f_dict['flair'] ]
            assert len(self.filenames) > 0, f"Unknown patient id {patid} !"
        if train:
            # remove empty images (zero-brain) for training
            self.filenames = [f_dict for f_dict in self.filenames if float(f_dict['brain_size']) > 0]
            # remove images without lesions for training
            self.filenames = [f_dict for f_dict in self.filenames if float(f_dict['lesion_size']) > 0]
        # select randomly a number of max_size slices (to save memory)
        if max_size is not None and max_size < len(self.filenames):
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:max_size]
        # if preload is true, load all images, labels, patids, sliceids and 
        # save in a list containing 4-tuples
        self.preload = preload
        self.preload_items = []
        if self.preload:
            for idx in range(len(self.filenames)):
                image, segm, patid, sliceid = self.load_item(idx)
                self.preload_items.append((image, segm, patid, sliceid))  # append as a tuple
        
    def read_csv_file(self, filename, separator=','):
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter=separator)
            filenames = []
            headers = ['flair', 'dwi', 't1', 't2', 'seg', 'lesion_size', 'brain_size']
            for line in reader:
                assert len(line) == len(headers)+1  # check if file is valid
                file_dict = dict()
                for idx, key in enumerate(headers):
                    file_dict[key] = line[idx].strip()
                filenames.append(file_dict)
        return filenames
    
    def read_image_to_tensor(self, filename, label=False):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if label:
            return torch.from_numpy(image).to(torch.uint8)
        return torch.from_numpy(image).float()
    
    def pat_id_from_path(self, filename_dict):
        return filename_dict["flair"].split("/")[0][-3:]

    def slice_id_from_path(self, filename_dict):
        return filename_dict["flair"].split("/")[-1].split(".")[0][-3:]
    
    
    def augmentation(self, ex_img, ex_seg):
        # Convert tensors to PIL Images
        ex_img = TF.to_pil_image(ex_img)
        ex_seg = TF.to_pil_image(ex_seg)

        # Apply horizontal flip with a 50% chance
        if random.random() > 0.5:
            ex_img = TF.hflip(ex_img)
            ex_seg = TF.hflip(ex_seg)

        # Apply a random rotation
        angle = random.uniform(-10, 10)
        ex_img = TF.rotate(ex_img, angle)
        ex_seg = TF.rotate(ex_seg, angle)

        # Convert PIL Images back to tensors
        aug_image = TF.to_tensor(ex_img)
        #aug_seg = TF.to_tensor(ex_seg)
        aug_seg = torch.from_numpy(np.array(ex_seg)).unsqueeze(0)

        return aug_image, aug_seg
    
    
    def load_item(self, item):
        # return a tuple of image, corresponding labels, patid and sliceid at the index 'item'
        # return the image (shape: 4xHxW), label (shape: 1xHxW), patid, sliceid
        dic = self.filenames[item]
        f = self.read_image_to_tensor(os.path.join(self.path, dic["flair"]))
        dwi = self.read_image_to_tensor(os.path.join(self.path, dic["dwi"]))
        t1 = self.read_image_to_tensor(os.path.join(self.path, dic["t1"]))
        t2 = self.read_image_to_tensor(os.path.join(self.path, dic["t2"]))
        img = torch.stack([f, dwi, t1, t2])
        pat_id = self.pat_id_from_path(dic)
        slice_id = self.slice_id_from_path(dic)
        if dic["seg"]:
            seg = self.read_image_to_tensor(os.path.join(self.path, dic["seg"]), label=True).unsqueeze(0)
            seg[seg > 0] = 1                                         
            return img, seg, pat_id, slice_id
        return img, pat_id, slice_id
    
    def __getitem__(self, item):
        # return a tuple of image, corresponding labels, patid and sliceid at the index 'item'
        if self.preload:
            return self.preload_items[item]
        aug = not self.val and bool(random.getrandbits(1))
        #aug = False
        if aug:
            ex_img, ex_seg, pat_id, slice_id = self.load_item(item)
            aug_image, aug_seg = self.augmentation(ex_img, ex_seg)
            return aug_image, aug_seg, pat_id, slice_id
        return self.load_item(item)
        
    def __len__(self):
        # return the number of elements in the dataset
        return len(self.filenames)


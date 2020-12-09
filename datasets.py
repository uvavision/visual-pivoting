import torch
from torch.utils.data import Dataset
import h5py
import tables
import json
import os
import numpy as np
from torch.distributions import normal, uniform
    
class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, languages, random = False, transform=None, cpi = 5):
        """
        :param data_folder: folder where data files are stored - /Users/skye/docs/image_dataset/dataset
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        

        # Open hdf5 file where images are stored
        self.random = random
        if self.random:
            m = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
            lfeats = m.sample((1,3*224*224)).squeeze(2).numpy() 
            self.imgs = lfeats
            
        else:
            img_data_name = data_name
            self.h = h5py.File(os.path.join(data_folder, self.split +  '_IMAGES_' + img_data_name + '.hdf5'), 'r')
            self.imgs = self.h['images']
            self.random = random
            self.languages = languages

        # Captions per images
        self.cpi = cpi
        print('captions per image is :', self.cpi)
        
        # Load encoded captions for source and target languages
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + languages[0] + '_' + data_name + '.json'), 'r') as j:
            self.captions_l1 = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + languages[0] + '_' + data_name + '.json'), 'r') as j:
            self.caplens_l1 = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + languages[1] + '_' + data_name + '.json'), 'r') as j:
            self.captions_l2 = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + languages[1] + '_' + data_name + '.json'), 'r') as j:
            self.caplens_l2 = json.load(j)
            
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions_l1)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
            
        if self.random:
            img = torch.from_numpy(self.imgs).view(3,224,224)
        else:
            img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        
        if self.transform is not None:
            img = self.transform(img)

        caption_l1 = torch.LongTensor(self.captions_l1[i])

        caplen_l1 = torch.LongTensor([self.caplens_l1[i]])

        caption_l2 = torch.LongTensor(self.captions_l2[i])

        caplen_l2 = torch.LongTensor([self.caplens_l2[i]])
        
        if self.split is 'TRAIN':
            
            return img, caption_l1, caplen_l1, caption_l2, caplen_l2
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions_l1 = torch.LongTensor(self.captions_l1[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            all_captions_l2 = torch.LongTensor(self.captions_l2[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption_l1, caplen_l1, all_captions_l1, caption_l2, caplen_l2, all_captions_l2

    def __len__(self):
        return self.dataset_size

    
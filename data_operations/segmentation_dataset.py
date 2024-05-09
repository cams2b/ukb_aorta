import os
import math
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable 

from data_operations.image_utils import channelwise_normalization


def prepare_single_image(image_path, pid):
    subject =tio.Subject(image=tio.ScalarImage(image_path), ID=pid)

    transform= tio.Compose([
            tio.Pad((1, 1, 0)),
            tio.ZNormalization(),
        ])
    
    dataset = tio.SubjectsDataset([subject], transform=transform)

    return dataset
    



def prepare_dataset(image_arr, mask_arr, num_classes=2, train=False):
    assert len(image_arr) == len(mask_arr), '[ERROR] Incorrect number of images and masks'
    subjects = []
    for img, msk in zip(image_arr, mask_arr):
        subject = tio.Subject(image=tio.ScalarImage(img), mask=tio.LabelMap(msk))
        subjects.append(subject)

    if train:
        ## try swapping padding and normalization ========================
        transform = tio.Compose([tio.ZNormalization(),
                                 tio.RandomMotion(p=0.2),
                                 tio.RandomBiasField(p=0.3),
                                 tio.RandomNoise(p=0.5),
                                 tio.RandomFlip(),
                                 tio.Pad((1, 1, 0)),
                                 tio.OneHot(num_classes=num_classes),
                                ])
    else:
        transform = tio.Compose([tio.ZNormalization(),
                                 tio.Pad((1, 1, 0)),
                                 tio.OneHot(num_classes=num_classes),
                                ])
    dataset = tio.SubjectsDataset(subjects, transform=transform)

    return dataset

#transform = tio.Compose([#tio.RandomMotion(p=0.2),
                                 #tio.Pad((1, 1, 0)),
                                 #tio.RandomBiasField(p=0.3),
                                 #tio.ZNormalization(),
                                 #tio.RandomNoise(p=0.5),
                                 #tio.RandomFlip(),
                                 #tio.OneHot(num_classes=num_classes),
                                #])

#transform = tio.Compose([
            #tio.Pad((1, 1, 0)),
            #tio.ZNormalization(),
            #tio.OneHot(num_classes=num_classes),
            #])


def patch_dataloader(dataset, patch_size, patches_per_image, label_probabilities, queue_length=300, num_workers=2):
    sampling_strategy = tio.data.LabelSampler(patch_size=patch_size, label_probabilities=label_probabilities)
    
    patch_generator = tio.Queue(
        subjects_dataset=dataset,
        max_length=queue_length,
        samples_per_volume=patches_per_image,
        sampler=sampling_strategy,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True
    ) 

    return patch_generator


def prepare_inference_dataset(image_arr, mask_arr, pid_arr=None):
    assert len(image_arr) == len(mask_arr), '[ERROR] Incorrect number of images and masks'

    subjects = []
    if pid_arr is None:
        for img, msk in zip(image_arr, mask_arr):
            subject = tio.Subject(image=tio.ScalarImage(img), mask=tio.LabelMap(msk))
            subjects.append(subject)
    else:
        for img, msk, pid in zip(image_arr, mask_arr, pid_arr):
            subject = tio.Subject(image=tio.ScalarImage(img), mask=tio.LabelMap(msk), ID=pid)
            subjects.append(subject)


    transform = tio.Compose([
        tio.Pad((2, 2, 0)),
        tio.ZNormalization(),
        tio.OneHot(),
    ])

    dataset = tio.SubjectsDataset(subjects, transform=transform)

    return dataset


class slice_dataset(Dataset):
    def __init__(self, dataset, transform=None, subset='train', slices_per_image=136, preload=True):
        self.dataset= dataset
        self.transform = transform
        self.subset = subset
        self.slices_per_image = slices_per_image
        self.slice_arr = []
        self.image = []
        self.mask = []
        self.steps_per_epoch = 0
        self.preload = preload
        if preload:
            self.preload_data()


    def initialize(self):
        if self.slices_per_image == None:
            print('[INFO] iterating dataset to calculate total slices')
            for i in self.dataset:
                img = i['image'][tio.DATA]
                self.steps_per_epoch += img.shape[-1]
                self.slice_arr.append(img.shape[-1])
            self.slices_per_image = int(math.floor(np.average(self.slice_arr)))
        else:
            self.steps_per_epoch = len(self.dataset) * self.slices_per_image


    def preload_data(self):
        steps_per_epoch = 0
        for i in range(len(self.dataset)):
            subject = self.dataset[i]
            x = subject['image'][tio.DATA]
            y = subject['mask'][tio.DATA]

            for i in range(x.shape[-1]):
                self.image.append(x[:, :, :, i].float())
                self.mask.append(y[:, :, :, i].float())
                steps_per_epoch += 1
        self.dataset = None
        
        self.steps_per_epoch = steps_per_epoch


    def __len__(self):
        ## this might have to be adjusted
        return self.steps_per_epoch
    
    def __getitem__(self, index):
        if self.preload:
            return self.image[index], self.mask[index]
        else:
            ## calculate image and slice
            img_idx = math.floor(index / self.slices_per_image)
            slice_idx = index % self.slices_per_image
            
            subject = self.dataset[img_idx]
            x = subject['image'][tio.DATA]
            y = subject['mask'][tio.DATA]
        
            x_slice = x[:, :, :, slice_idx].float()
            y_slice = y[:, :, :, slice_idx].float()
            x_slice, y_slice = Variable(x_slice), Variable(y_slice)

            return x_slice, y_slice
        
        
        


class dataset_2d(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_length = len(dataset)

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index):
        subject = self.dataset[index]
        x = subject['image'][tio.DATA]
        y = subject['mask'][tio.DATA]

        x = torch.squeeze(x, dim=-1)
        y = torch.squeeze(y, dim=-1)


        return x, y
        


class dataset_3d(Dataset):
    def __init__(self, x, y, pid, z_score_channelwise=False, inference_mode=False):
        self.x = x
        self.y = y
        self.pid = pid
        self.z_score_channelwise = z_score_channelwise
        self.inference_mode = inference_mode
        self.data_length = len(self.x)

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index):        
        id = self.pid[index]
        itk_image = sitk.ReadImage(self.x[index])
        itk_image = channelwise_normalization(itk_image)

        if self.inference_mode:
            return itk_image, id
        else:
            itk_mask = sitk.ReadImage(self.y[index])
            return itk_image, itk_mask, id




## ========================== inferencing code ==================================
def prepare_3d_for_2d(subject, num_classes=2):

    transform = tio.Compose([
            tio.Pad((1, 1, 0)),
            tio.OneHot(num_classes=num_classes),
        ])
    
    dataset = tio.SubjectsDataset([subject], transform=transform)

    return dataset


def prepare_3d_for_2d_inferencing(subject, num_classes=2):

    transform = tio.Compose([
            #tio.Pad((1, 1, 0)),
            tio.OneHot(num_classes=num_classes),
        ])
    
    dataset = tio.SubjectsDataset([subject], transform=transform)

    return dataset


def process_slice_inferencing(slice_arr, num_classes=2):
    """
    Convert itk image slices into a subject dataset for inferencing
    :slice_arr: a list of (x, y, 1) itk images from a 3-dimensional itk_image
    :num_classes: the number of possible classes for output (this is likely not necessary)
    """
    subjects = []
    for slice in zip(slice_arr):
        slice = slice[0]
        subject = tio.Subject(image=tio.ScalarImage.from_sitk(slice))
        subjects.append(subject)

    transform = tio.Compose([tio.ZNormalization(),
                                tio.Pad((1, 1, 0)),
                                tio.OneHot(num_classes=num_classes),
                            ])
    dataset = tio.SubjectsDataset(subjects, transform=transform)

    return dataset



def process_slice_mask_inferencing(slice_arr, num_classes=2):
    """
    Convert itk image slices into a subject dataset for inferencing
    :slice_arr: a list of (x, y, 1) itk images from a 3-dimensional itk_image
    :num_classes: the number of possible classes for output (this is likely not necessary)
    """
    subjects = []
    for slice in zip(slice_arr):
        slice = slice[0]
        subject = tio.Subject(image=tio.ScalarImage.from_sitk(slice))
        subjects.append(subject)

    transform = tio.Compose([tio.ZNormalization(),
                                tio.Pad((1, 1, 0)),
                            ])
    dataset = tio.SubjectsDataset(subjects, transform=transform)

    return dataset
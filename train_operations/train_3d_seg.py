import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


from config import config
from data_operations.utils import process_excel
from data_operations.segmentation_dataset import *
from train_operations.loss_functions import *
from train_operations.network_trainer import network_trainer
from train_operations.make_experiment import *
from models.unet import unet
from models.slice_unet import *

def perform_training():
    print('[INFO] 3D image segmentation training')
    make_experiment()

    train_images, train_masks = process_excel(config.train_path)
    validation_images, validation_masks = process_excel(config.validation_path)
    train_dataset = prepare_dataset(train_images, train_masks, train=True)
    validation_dataset = prepare_dataset(validation_images, validation_masks, train=False)

    train_patch_generator = patch_dataloader(train_dataset, 
                                            config.patch_size,
                                            config.patches_per_image, 
                                            config.label_probabilities, 
                                            queue_length=config.queue_length, 
                                            num_workers=2)
    
    validation_patch_generator = patch_dataloader(validation_dataset,
                                                  config.patch_size,
                                                  config.patches_per_image,
                                                  config.label_probabilities,
                                                  queue_length=config.queue_length,
                                                  num_workers=2)
    
    train_dataloader = torch.utils.data.DataLoader(train_patch_generator, batch_size=config.batch_size, pin_memory=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_patch_generator, batch_size=config.batch_size, pin_memory=True)

    if config.resume_training:
        model = torch.load(config.weight_path)
    else:
        if config.use_slice:
            model = slice_unet(1, 1)
        else:
            model = unet(1, 1)
            
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    criterion = GDiceLoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=config.lr_patience)


       

    net_train = network_trainer(train_dataloader=train_dataloader,
                                validation_dataloader=validation_dataloader,
                                model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                scheduler=scheduler,
                                epochs=config.epochs)
        
    net_train.run()
    

    
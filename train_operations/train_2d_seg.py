import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


from config import config
from data_operations.utils import process_excel
from data_operations.segmentation_dataset import *
from train_operations.loss_functions import *
from train_operations.network_trainer_2d import network_trainer_2d
from train_operations.make_experiment import *

## models
from models.slice_unet import *
from models.unet_plus_plus import *

def perform_training():
    print('[INFO] 2D image segmentation training')
    make_experiment()
    torch.backends.cudnn.benchmark = True
    
    train_images, train_masks = process_excel(config.train_path)
    validation_images, validation_masks = process_excel(config.validation_path)


    train_data = prepare_dataset(train_images, train_masks, num_classes=2, train=False)
    validation_data = prepare_dataset(validation_images, validation_masks, num_classes=2, train=False)

    train_dataset = dataset_2d(train_data)
    validation_dataset = dataset_2d(validation_data)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=3, pin_memory=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=3, pin_memory=True)
    
    if config.resume_training:
        model = torch.load(config.weight_path)
 
    model = unet_plus_plus(1, 1)
    
    if torch.cuda.is_available():
        print('[INFO] Running model on GPU')
        model.cuda()
    else:
        model.cpu()
    
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    criterion = GDiceLoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=config.lr_patience)


       

    net_train = network_trainer_2d(train_dataloader=train_dataloader,
                                   validation_dataloader=validation_dataloader,
                                   model=model,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   scheduler=scheduler,
                                   epochs=config.epochs)
        
    net_train.run()
import pandas as pd
import torch
import torchio as tio
from torchmetrics.classification import Dice

from train_operations.loss_functions import *
from data_operations.image_utils import*
from config import config


class network_trainer_2d(object):
    def __init__(self, train_dataloader=None, validation_dataloader=None, model=None, optimizer=None, criterion=None, scheduler=None, epochs=None, record_data=True):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs
        self.record = record_data
        self.metric = Dice(average='macro', num_classes=2).cuda()
        if record_data:
            self.init_data_recorder()

    def init_data_recorder(self):
        self.data_recorder = pd.DataFrame()
        self.train_loss = []
        self.validation_loss = []
        self.train_metric = []
        self.validation_metric = []

    def train(self, epoch):
        print('[INFO] Train loop')
        use_cuda = torch.cuda.is_available()
        self.model.train()
        n_processed = 0
        training_loss = 0.0
        running_corrects = 0

        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            if use_cuda:        
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            y_pred = torch.softmax(y_pred, dim=1)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            
            dsc = self.metric(torch.argmax(y_pred, dim=1), torch.argmax(y, dim=1))
            n_processed += 1
            running_corrects += dsc
            training_loss += loss.data.item()
            avg_loss = training_loss / n_processed

            dsc = running_corrects / n_processed
            
            print('Train Epoch: {}({}/{})]\tLoss: {:.6f}\t multi_class_dice_coef: {:.6f}'.format(epoch, n_processed, len(self.train_dataloader), avg_loss, dsc))
        
        return avg_loss, dsc


    def test(self):
        use_cuda = torch.cuda.is_available()
        self.model.eval()
        n_processed = 0
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for (x, y) in self.validation_dataloader:
                
                if use_cuda:
                    x, y = x.cuda(), y.cuda()

                
                y_pred = self.model(x)
                y_pred = torch.softmax(y_pred, dim=1)

                test_loss += self.criterion(y_pred, y)

                dsc = self.metric(torch.argmax(y_pred, dim=1), torch.argmax(y, dim=1))
                n_processed += 1
                correct += dsc


        test_loss /= len(self.validation_dataloader)  # loss function already averages over batch size

        avg_dsc = 100. * correct / n_processed
        print('\nTest set: Average loss: {:.4f}, multi_class_dice_coef: ({:.0f}%)\n'.format(test_loss, avg_dsc))
        return test_loss, avg_dsc.item()



    def train_test(self, epoch):
        train_loss, train_metric = self.train(epoch)
        val_loss, val_metric = self.test()
        if self.record:
            self.train_loss.append(train_loss)
            self.train_metric.append(train_metric)
            self.validation_loss.append(val_loss)
            self.validation_metric.append(val_metric)

        return val_loss, val_metric


    def run(self):
        print('[INFO] begin training..')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        weight_path = config.output_path + config.experiment_name + '/' + config.model_name + '/weights/'
        assert os.path.exists(weight_path), '[ERROR] the weight path provided does not exist'


        validation_metric = 0
        for epoch in range(0, self.epochs):
            val_loss, current_metric = self.train_test(epoch)
            self.scheduler.step(val_loss)
            if current_metric > validation_metric:
                validation_metric = current_metric
                torch.save(self.model, weight_path + '_epoch_' + str(epoch) + '_val_' + str(round(current_metric, 3)) + '_last_weights.pth')
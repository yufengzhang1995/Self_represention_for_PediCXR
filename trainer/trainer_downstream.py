import torch
from torch.nn.functional import softmax
from abc import abstractmethod
import numpy as np
import metric as module_metric
import logger_utils as module_log
import loss_trams as module_loss
import pandas as pd
import glob, os, copy

from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

from utils.confusion_utils import cross_entropy_over_annotators,confusion_matrix_estimators

# Class Inheritance of trainer_trams.py
# |--BaseTrainer
#      |--Trainer
#          |--TRAM_Trainer
#          |--TRAM_Ordinal_Trainer
#          |--Evaluation_Trainer

class BaseTrainer:
    """
    Base class for all trainers.
    Setup the training logic with early stopping, model saving, logging, etc.
    Inputs:
        save_dir: directory to save model
        logger: logger for logging information
        num_epochs: number of epochs to train
        save_period: number of epochs to save model
        early_stop: number of epochs to wait before early stopping
        monitor_mode: mode to monitor model performance, 'min' or 'max' with the target metric
    """
    def __init__(self, save_dir, logger, num_epochs, save_period, monitor_mode, early_stop=0, fold=-1):

        self.epochs = int(num_epochs)
        self.start_epoch = 1

        self.monitor_on = int(early_stop) > 0
        # configuration to monitor model performance and save best
        if self.monitor_on:
            self.mnt_mode, self.mnt_metric = monitor_mode.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = int(early_stop)
        else:
            self.mnt_mode = 'off'
            self.mnt_best = 0

        self.logger = logger
        self.metric_tracker = module_metric.FullMetricTracker()

        self.save_dir = save_dir
        self.save_period = int(save_period)
        self.fold = fold
        
        
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
    @abstractmethod
    def _save_checkpoint(self, epoch, save_best=False):
        raise NotImplementedError
    
    def train(self):
        """
        Full training logic with early stopping and model saving
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # The result is a dictory containing training loss and metrics, validation loss and metrics
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            self.metric_tracker.update(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            self._save_checkpoint(epoch, save_best=best)
            self._save_metric_tracker()

    def _save_metric_tracker(self):
        """
        Saving metric tracker
        """
        filename = str(self.save_dir / 'metric_tracker.csv') if self.fold == -1 \
            else str(self.save_dir / 'metric_tracker_{}.csv'.format(self.fold))
        self.metric_tracker.get_data().to_csv(filename, index=False)
        self.logger.info("Saving metric tracker: {} ...".format(filename))

class Trainer(BaseTrainer):
    """
    Trainer class
    Inputs:
        config <ConfigParser instance>: containing all the configurations
        model <torch.nn.Module>: the model to be trained
        optimizer <torch.optim.Optimizer>: optimizer for the model
        logger <logging.Logger>: logger for logging information
        device <torch.device>: device to run the model
        train_loader <torch.utils.data.DataLoader>: training data loader
        val_loader <torch.utils.data.DataLoader>: validation data loader
        augmentation <nn.Module>: augmentation for training data
        lr_scheduler <torch.optim.lr_scheduler._LRScheduler>: learning rate scheduler   
    """
    def __init__(self, config, model, optimizer, logger,
                device, train_loader, val_loader, augmentation=None, 
                lr_scheduler=None, fold=-1):
        
        cfg_trainer = config['trainer']
        self.config = config

        # This super().__init__() will call the __init__() of the BaseTrainer class, 
        # which will initialize the following variables used in the Trainer class:
        # self.logger and self.save_dir among others mentioned in cfg_trainer

        super().__init__(config.save_dir, logger, cfg_trainer['epochs'], 
                         cfg_trainer['save_period'], cfg_trainer['monitor'], 
                         cfg_trainer['early_stop'], fold=fold)

        # obtain the log directory from the config instance
        self.log_dir = config.log_dir

        # Setup the device which to run the model
        self.device = device

        # Initialize the metrics tracker
        self.metric_ftns = config['metrics']

        # Initilzate the tensorboard writer 
        self.writer = module_log.get_tensorboard_writer(
            self.log_dir, logger, cfg_trainer['tensorboard'])

        self.train_metrics = module_metric.MetricTracker(
            'loss', *[m for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = module_metric.MetricTracker(
            'loss', *[m for m in self.metric_ftns], writer=self.writer)

        # Initialize the optimizr and learning rate scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.augmentation = augmentation.to(self.device) if augmentation else None

        # Initialize the network as self.model
        self.model = model

        # Initialize the loss function
        self.criterion = self._init_loss()

        # Setup the training and validation data loader
        self.data_loader = train_loader
        self.val_data_loader = val_loader

        # Initilzate the tensorboard writer 
        self.writer = module_log.get_tensorboard_writer(
            self.log_dir, logger, cfg_trainer['tensorboard'])
    
    @abstractmethod
    def _init_loss(self):
        """
        Initialize the loss function
        """
        raise NotImplementedError
    
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        raise NotImplementedError

    def _model_sanity_check(self):
        pass
        
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        Input:
            epoch <python int>: current training epoch.
        Output:
            log <python dict>: a log that contains average loss and metric in this epoch.
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():

            # Initialize the output and target tensor in a validation 
            # epoch for metric calculation
            output_epoch = torch.tensor([]).to(self.device)
            target_epoch = torch.tensor([]).to(self.device) 

            for batch_idx, data in enumerate(self.val_data_loader):

                target, image = data['lab'].to(self.device), data['img'].to(self.device)

                output = self.model(image)
                
                target_epoch = torch.cat((target_epoch, target))
                output_epoch = torch.cat((output_epoch, output))

                # loss = self.criterion(softmax(output, dim=1), target)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.val_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

            # Calculate the metric for the validation epoch
            for met in self.metric_ftns:
                self.valid_metrics.update(met, getattr(module_metric, met)(
                    output_epoch, target_epoch, self.device))
        
        if epoch == 1:
            self._model_sanity_check()
               
        return self.valid_metrics.result()
    
    def test(self, test_loader=None):
        """
        Evaluate the best model on test dataset
        """

        # Use the best recorded or latest model for testing 
        file_name = 'model_best.pth' if self.fold == -1 else f'model_best_{self.fold}.pth'
        best_path = self.save_dir / file_name
        if best_path.is_file():
            self._restore_checkpoint(str(best_path))
        else:
            self._restore_checkpoint(self._get_the_lastest_checkpoint(self.save_dir))
            
        self.model.eval()

        test_metrics = module_metric.MetricTracker('loss', *[m for m in self.metric_ftns])
        test_metrics.reset()

        with torch.no_grad():

            # Initialize the output and target tensor in a validation epoch for metric calculation
            output_epoch = torch.tensor([]).to(self.device)
            target_epoch = torch.tensor([]).to(self.device)
            
            # Initialized the embedding tensor, grade and uncertainty tensors for visualization 
            embedding_epoch = torch.tensor([]).to(self.device)
            uncertainty_epoch = torch.tensor([])
            grade_epoch = torch.tensor([])
            idx_epoch = torch.tensor([])

            for data in test_loader:

                target, image = data['lab'].to(self.device), data['img'].to(self.device)

                embedding, output = self.model(image, get_embedding=True)

                target_epoch = torch.cat((target_epoch, target))
                output_epoch = torch.cat((output_epoch, output))

                embedding_epoch = torch.cat((embedding_epoch, embedding))
                grade_epoch = torch.cat((grade_epoch, data['raw_grades']))
                uncertainty_epoch = torch.cat((uncertainty_epoch, data['uncertainty']))
                idx_epoch = torch.cat((idx_epoch, data['idx']))

                # loss = self.criterion(softmax(output, dim=1), target)
                loss = self.criterion(output, target)

                test_metrics.update('loss', loss.item())

            # Calculate the metric for the validation epoch
            for met in self.metric_ftns:
                test_metrics.update(met, getattr(module_metric, met)(
                    output_epoch, target_epoch, self.device))


        save_name = 'test.csv' if self.fold == -1 else f'test_{self.fold}.csv'
        self.save_metrics(test_metrics.result(), save_name)

        # save the target_epoch and output_epoc, the embedding, grade and uncertainty for visualization
        save_name = 'test_visualization.npy' if self.fold == -1 else f'test_visualization_{self.fold}.npy'

        with open(self.log_dir / save_name, 'wb') as fild_handle:
            np.save(fild_handle, target_epoch.cpu().numpy())
            np.save(fild_handle, output_epoch.cpu().numpy())
            np.save(fild_handle, embedding_epoch.cpu().numpy())
            np.save(fild_handle, grade_epoch.cpu().numpy())
            np.save(fild_handle, uncertainty_epoch.cpu().numpy())
            np.save(fild_handle, idx_epoch.cpu().numpy())


    def save_metrics(self, metrics_dict, filename, col_name=None):
        """
        Save metrics information to a csv file in the path
        """
        df_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=[col_name])
        df_metrics.to_csv(self.log_dir / filename)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_state_dict = self.model.module.state_dict() if hasattr(
            self.model, 'module') else self.model.state_dict()
        
        state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict()
        }

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        checkpoint_str = f'checkpoint-epoch{epoch}.pth' if self.fold == -1 \
            else f'checkpoint-epoch{epoch}_{self.fold}.pth'
        filename = str(self.save_dir / checkpoint_str)
        if epoch % self.save_period == 0:
            torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            file_name = 'model_best.pth' if self.fold == -1 else f'model_best_{self.fold}.pth'
            best_path = str(self.save_dir / file_name)
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)
    
    def _restore_checkpoint(self, checkpoint):
        """
        Load model(s) from saved checkpoint
        """
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location='cpu')
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.info("Checkpoint loaded.")

    def _get_the_lastest_checkpoint(self, dir):
        """
        Get the lastest checkpoint from the directory
        """
        list_of_files = glob.glob(str(dir) + '/*.pth')
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file


class TRAM_Trainer(Trainer):
    """
    Trainer class for the TRAM model
    """
    def __init__(self, *kwags, **kwargs):
        super().__init__(*kwags, **kwargs)
        self.train_metrics = module_metric.MetricTracker(
            'loss', 'base_loss', *[m for m in self.metric_ftns], 
            'pi_loss',*[f'pi_{m}' for m in self.metric_ftns], writer=self.writer)
        
    def _init_loss(self):
        """
        Initialize the loss function
        """
        return self.config.init_obj('loss', module_loss)
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch in TRAM
        Input:
            epoch <python int>: current training epoch.
        Output:
            log <python dict>: a log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):

            target, image = data['lab'].to(self.device), data['img'].to(self.device)
            add_data = data['add'].to(self.device)

            self.optimizer.zero_grad()
            
            if self.augmentation is not None:
                with torch.no_grad():
                    image = self.augmentation(image)

            output, add_output = self.model(image, add_data, train=True)

            loss = self.criterion(output, target, add_output, train=True)
                       
            loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            self.train_metrics.update('loss', loss.item())

            self.train_metrics.update('base_loss', self.criterion.base_loss.item())
            self.train_metrics.update('pi_loss', self.criterion.pi_loss.item())

            for met in self.metric_ftns:
                
                self.train_metrics.update(met, getattr(module_metric, met)(
                    output, target, self.device))
                
                self.train_metrics.update(f'pi_{met}', getattr(module_metric, met)(
                    add_output, target, self.device)) 

            if batch_idx % 10 == 0: #int(len(self.data_loader)/10)
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

        log = self.train_metrics.result()

        if self.val_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log        


            

class Evaluation_Trainer(Trainer):
    """
    Trainer class for the evaluation of the encoder either with fine-tuning or linear probing.
    The sanity check is performed in this class.
    """
    def __init__(self, *kwags, **kwargs):
        super().__init__(*kwags, **kwargs)
        # should the encoder be frozen
        self.freeze = self.config['encoder']['args']['freeze']

        if self.freeze: # get a deep copy of the encoder
            self.ref_encoder = copy.deepcopy(self.model.module.encoder) if hasattr(
                self.model, 'module') else copy.deepcopy(self.model.encoder)
    
    def _init_loss(self):
        """
        Initialize the cross_entropy loss function
        """
        return torch.nn.CrossEntropyLoss()

    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch in linear probing or fine-tuning
        """
        self.model.train()
        if self.freeze:
            self.model.encoder.eval()

        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):

            target, image = data['lab'].to(self.device), data['img'].to(self.device)

            if self.augmentation is not None:
                with torch.no_grad():
                    image = self.augmentation(image)
                    
            self.optimizer.zero_grad()

            output = self.model(image, freeze_encoder=self.freeze)
            loss = self.criterion(softmax(output, dim=1), target)
            loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                
                self.train_metrics.update(met, getattr(module_metric, met)(
                    output, target, self.device))

            if batch_idx % 10 == 0: #int(len(self.data_loader)/10)
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

        log = self.train_metrics.result()

        if self.val_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log        
    
    def _model_sanity_check(self):
        if self.freeze:
            # get the state dict of self.ref_encoder
            ref_state_dict = self.ref_encoder.state_dict()
            # get the state dict of the encoder of the model
            if hasattr(self.model, 'module'):
                model_state_dict = self.model.module.encoder.state_dict()
            model_state_dict = self.model.encoder.state_dict()
            # check if the state dict of the encoder of the model is equal to the state dict of self.ref_encoder
            for k in ref_state_dict:
                if 'op_threshs' not in k:
                    assert torch.equal(ref_state_dict[k], model_state_dict[k]), \
                        "The encoder is not properly frozen on the {} item".format(k)  
            self.logger.info("The encoder is properly frozen")

class Confusion_Trainer(Trainer):
    """
    Trainer class for the evaluation of the encoder either with fine-tuning or linear probing.
    The sanity check is performed in this class.
    """
    def __init__(self, confusion_matrices,  *kwags, **kwargs):
        super().__init__(*kwags, **kwargs)
        # should the encoder be frozen
        self.freeze = self.config['encoder']['args']['freeze']

        if self.freeze: # get a deep copy of the encoder
            self.ref_encoder = copy.deepcopy(self.model.module.encoder) if hasattr(
                self.model, 'module') else copy.deepcopy(self.model.encoder)

        # confusion matrices of annotators
        self.confusion_matrices = confusion_matrices
        
        self.scale = 0.01

    def _init_loss(self):
        """
        Initialize the cross_entropy loss function
        """
        return torch.nn.CrossEntropyLoss()


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch in linear probing or fine-tuning
        """
        self.model.train()
        if self.freeze:
            self.model.encoder.eval()

        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):

            target, image = data['lab'].to(self.device), data['img'].to(self.device)

            reviewer_label = data['reviewer_label'].to(self.device)

            if self.augmentation is not None:
                with torch.no_grad():
                    image = self.augmentation(image)
                    
            self.optimizer.zero_grad()

            output = self.model(image, freeze_encoder=self.freeze)
            # loss = self.criterion(softmax(output, dim=1), target)

            # 1. weighted cross-entropy
            weighted_cross_entropy = cross_entropy_over_annotators(reviewer_label, 
                                        softmax(output, dim=1), self.confusion_matrices)

            # 2. trace of confusion matrices:
            stack_traces = torch.stack([torch.trace(
                self.confusion_matrices[I, :, :]) for I in range(
                self.confusion_matrices.shape[0])])
            trace_norm = torch.mean(stack_traces)

            loss = weighted_cross_entropy + self.scale * trace_norm
            loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                
                self.train_metrics.update(met, getattr(module_metric, met)(
                    output, target, self.device))

            if batch_idx % 10 == 0: #int(len(self.data_loader)/10)
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.logger.debug(self.confusion_matrices[0, :, :])

        log = self.train_metrics.result()

        if self.val_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log   
    
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_state_dict = self.model.module.state_dict() if hasattr(
            self.model, 'module') else self.model.state_dict()
        # TODO: check the multi-gpu compatibility for confusion matrices
        state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'confusion_matrices': self.confusion_matrices
        }

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        checkpoint_str = f'checkpoint-epoch{epoch}.pth' if self.fold == -1 \
            else f'checkpoint-epoch{epoch}_{self.fold}.pth'
        filename = str(self.save_dir / checkpoint_str)
        if epoch % self.save_period == 0:
            torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            file_name = 'model_best.pth' if self.fold == -1 else f'model_best_{self.fold}.pth'
            best_path = str(self.save_dir / file_name)
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
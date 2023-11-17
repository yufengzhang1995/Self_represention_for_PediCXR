import torch
from abc import abstractmethod
from numpy import inf
import utils.ddp_utils as ddp_util_module
import metric as module_metric
import utils.logger_utils as module_log
import torch.nn.functional as torch_functional
import math
import utils.main_utils as util_module


# Class Inheritance of trainer_ssl.py
# |--SSL_BaseTrainer
#      |--SSL_Trainer
#          |--BYOL_Trainer

class SSL_BaseTrainer:
    """
    Base class for all self-supervised learning (SSL) trainers.
    Setup the training logic with early stopping, model saving, logging, etc.
    Inputs:
        logger: logger for logging information
        num_epochs: number of epochs to train
        save_period: number of epochs to save model
        monitor_mode: mode to monitor model performance, 'min' or 'max' with the target metric
        early_stop: number of epochs to wait before early stopping
        start_epoch: the epoch number to start training
        lr_scheduler: learning rate scheduler
    """
    def __init__(self, save_dir, logger, num_epochs, save_period, 
                 monitor_mode, start_epoch=0, early_stop=0):

        self.epochs = int(num_epochs)
        self.start_epoch = int(start_epoch)
        
        self.monitor_on = int(early_stop) > 0
        # configuration to monitor model performance and save best
        if self.monitor_on:
            self.mnt_mode, self.mnt_metric = monitor_mode.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = int(early_stop)
        else:
            self.mnt_mode = 'off'
            self.mnt_best = 0

        self.logger = logger
        self.metric_tracker = module_metric.FullMetricTracker()

        self.save_dir = save_dir
        self.save_period = int(save_period)

    
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic with early stopping and model saving
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # The result is a dictory containing training loss and metrics, 
            # validation loss and metrics
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            self.metric_tracker.update(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, 
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not based on mnt_metric
                    improved = (self.mnt_mode == 'min' and log[
                        self.mnt_metric] <= self.mnt_best) or (self.mnt_mode == 'max' and log[
                        self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is " + \
                                        "disabled.".format(self.mnt_metric))
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

    @abstractmethod
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        """
        raise NotImplementedError
    
    def _save_metric_tracker(self):
        """
        Saving metric tracker
        """
        filename = str(self.save_dir / 'metric_tracker.csv')
        if util_module.is_main_process():
            self.metric_tracker.get_data().to_csv(filename, index=False)
        self.logger.info(f"Saving metric tracker ...")

class SSL_Trainer(SSL_BaseTrainer):
    """
    Trainer class for the SSL model
    Inputs:
        config <ConfigParser instance>: containing all the configurations
        learner <torch.nn.Module>: the ssl learner model
        device <torch.device>: device to run the model
        train_loader <torch.utils.data.DataLoader>: training data loader
        val_loader <torch.utils.data.DataLoader>: validation data loader
        feature_func <python string>: name of the feature function to 
            extract features from the encoder
        start_epoch <python int>: the epoch number to start training
        lr_scheduler <torch.optim.lr_scheduler>: learning rate scheduler
        distributed <python bool>: if True, use distributed training
    """
    def __init__(self, config, learner, optimizer, device, train_loader, val_loader, logger,
                 feature_func='features', start_epoch=1, lr_scheduler=None, distributed=False):

        self.cfg_trainer = config['trainer']

        # This super().__init__() will call the __init__() of the SSL_BaseTrainer class, 
        # which will initialize the following variables used in the BYOL_Trainer class:
        # self.logger, self.learner, self.optimizer, self.save_dir and self.scheduler,
        # among the rest mentioned in self.cfg_trainer.
        super().__init__(config.save_dir, logger, self.cfg_trainer['epochs'], 
                         self.cfg_trainer['save_period'], self.cfg_trainer['monitor'], 
                         start_epoch, self.cfg_trainer['early_stop'])
        
        self.learner = learner
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # obtain the log directory from the config instance
        self.log_dir = config.log_dir

        # Setup the device which to run the model
        self.device = device

        # Setup the distributed training
        self.distributed = distributed

        # Setup the training and validation data loader
        self.data_loader = train_loader
        self.val_data_loader = val_loader

        # Setup the feature function to extract features from the encoder
        self.feature_func = feature_func
    
        # Initilzate the tensorboard writer 
        self.writer = module_log.get_tensorboard_writer(
            self.log_dir, logger, self.cfg_trainer['tensorboard'], distributed)

        # Initialize the metrics tracker
        self.self_define_metric = ['average_std', 'reference_std']
        self.train_metrics = module_metric.MetricTracker('loss', *[m for m in self.self_define_metric], 
                                                         writer=self.writer)
        self.valid_metrics = module_metric.MetricTracker('loss', writer=self.writer)
    
    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        Input:
            epoch <python int>: current training epoch.
        Output:
            log <python dict>: a log that contains information about validation
        """
        if self.distributed:
            self.val_data_loader.sampler.set_epoch(epoch)

        self.learner.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for _, data in enumerate(self.val_data_loader):
                image = data['img'].to(self.device)

                ssl_loss = self.learner(image)
                loss = ssl_loss.mean()
                self.valid_metrics.update('loss', loss.item())

        return self.valid_metrics.result()
    

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        Inputs:
            epoch <python int>: current epoch number
            save_best <python bool>: if True, rename the saved checkpoint to 
                'checkpoint_best.pth' and save to the desired location

        The self.learner, self.optimizer, and self.lr_scheduler are saved in the checkpoint
        The self.learner should have the net attribute, which is the backbone network
            function get_embedding_dimension(), get_drop_layer() and get_image_size() 
            should be implemented in the self.learner
        """
        # prepare the checkpoint dictionary for resuming training
        model_state_dict = self.learner.module.state_dict() if hasattr(
            self.learner, 'module') else self.learner.state_dict()
        checkpoint = {'epoch': epoch,
            'learner': model_state_dict,
            'optimizer': self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        # prepare the backbone dictionary for down-stream tasks
        save_target = self.learner.module if hasattr(self.learner, 'module') else self.learner
        backbone = save_target.net
        backbone_state_dict = backbone.module.state_dict() if hasattr(
            backbone, 'module') else backbone.state_dict()
        backbone_checkpoint = {'backbone': backbone_state_dict,
                               'feature_dim': save_target.get_embedding_dimension(),
                               'layer_drop':save_target.get_drop_layer(),
                               'image_size': save_target.get_image_size()}
        
        # save the checkpoint
        filename = str(self.save_dir / f'checkpoint-epoch{epoch}.pth')
        backbone_filename = str(self.save_dir / f'backbone-epoch{epoch}.pth')
        if epoch % self.save_period == 0:
            util_module.save_on_master(checkpoint, filename)
            util_module.save_on_master(backbone_checkpoint, backbone_filename)
            self.logger.info(f"Saving checkpoint on epoch {epoch} ...")

        # save the best checkpoint
        if save_best:
            best_path = str(self.save_dir / 'checkpoint_best.pth')
            best_backbone_path = str(self.save_dir / 'backbone_best.pth')
            util_module.save_on_master(checkpoint, best_path)
            util_module.save_on_master(backbone_checkpoint, best_backbone_path)
            self.logger.info("Saving current best: checkpoint_best.pth ...")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)
    

class BYOL_Trainer(SSL_Trainer):

    def __init__(self, *kwags, **kwargs):
        super().__init__(*kwags, **kwargs)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        Input:
            epoch <python int>: current training epoch.
        Output:
            log <python dict>: a log that contains average loss and metric in this epoch.
        """
        if self.distributed:
            self.data_loader.sampler.set_epoch(epoch)

        self.learner.train(True)
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):

            image = data['img'].to(self.device)

            self.optimizer.zero_grad()
            ssl_loss = self.learner(image)
            loss = ssl_loss.mean()
            loss.backward()

            self.optimizer.step()
            if hasattr(self.learner, 'module'):
                self.learner.module.update_moving_average()
            else:
                self.learner.update_moving_average()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.train_metrics.update('loss', loss.item())

            with torch.no_grad():
                # size of embedding is (batch_size, embedding_dim=2048)
                if hasattr(self.learner, 'module'):
                    embedding = getattr(self.learner.module, self.feature_func)(image)
                else:
                    embedding = getattr(self.learner, self.feature_func)(image)
                l2_normalized = torch_functional.normalize(embedding, p=2, dim=-1)
                avg_std = torch.mean(torch.std(l2_normalized, dim=-1))

            self.train_metrics.update('average_std', avg_std.item())
            self.train_metrics.update('reference_std', 1 / math.sqrt(embedding.shape[1]))

            if len(self.data_loader) > 20:
                if batch_idx % (int(len(self.data_loader)/20)) == 0: #
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
    
class DINO_Trainer(SSL_BaseTrainer):
    """
    Trainer class for the DINO model
    Inputs:
        config <python dict>: Dict containing configurations, hyperparameters for training,
            contents of `config.json` file for example.
        student <torch.nn.Module>: The student model to be trained.
        teacher <torch.nn.Module>: The teacher model to be trained.
        optimizer <torch.optim>: The optimizer used for training.
        loss_func <torch.nn.Module>: The loss function used for training.
        augmentation <torch.nn.Module>: The augmentation used for training.
        train_loader <torch.utils.data.DataLoader>: The dataloader for training.
        val_loader <torch.utils.data.DataLoader>: The dataloader for validation.
        logger <logging.Logger>: The logger instance.
        start_epoch <python int>: The starting epoch number.
        distributed <python bool>: Whether to use distributed training.
        lr_scheduler <np.array>: an array of scheduled learning rate.
        fp16_scaler <torch.cuda.amp.GradScaler>: The mix-precision scaler.
        wd_scheduler <np.array>: an array of scheduled weight decay.
        momentum_scheduler <np.array>: an array of scheduled momentum.
    """
    def __init__(self, config, student, teacher, teacher_without_ddp, 
                 optimizer, loss_func, augmentation,
                 train_loader, val_loader, logger, 
                 start_epoch=1, distributed=True, lr_scheduler=None, 
                 fp16_scaler=None, wd_scheduler=None, momentum_scheduler=None):
        
        self.cfg_trainer = config['trainer']
        super().__init__(config.save_dir, logger, self.cfg_trainer['epochs'], 
                         self.cfg_trainer['save_period'], self.cfg_trainer['monitor'], 
                         start_epoch, self.cfg_trainer['early_stop'])
        
        # initialize the model, optimizer and loss
        self.student = student
        self.teacher = teacher
        self.teacher_without_ddp = teacher_without_ddp
        self.optimizer = optimizer
        self.dino_loss = loss_func
        self.augmentation = augmentation
        # obtain the log directory from the config instance
        self.log_dir = config.log_dir

        # Setup the distributed training
        self.distributed = distributed

        # Setup the training and validation data loader
        self.data_loader = train_loader
        self.val_data_loader = val_loader

        # Initilzate the tensorboard writer
        self.writer = module_log.get_tensorboard_writer(
            self.log_dir, logger, self.cfg_trainer['tensorboard'], distributed)

        # Initialize the fp16 scaler and scheduler
        self.fp16_scaler = fp16_scaler
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.momentum_scheduler = momentum_scheduler

        # Some other training setting
        self.clip_grad = self.cfg_trainer['clip_grad']
        self.freeze_last_layer = self.cfg_trainer['freeze_last_layer']

    def _train_epoch(self, epoch):
        if self.distributed:
            self.data_loader.sampler.set_epoch(epoch)

        metric_logger = ddp_util_module.MetricLogger(delimiter="  ", 
                                writer=self.writer, logger=self.logger)
        header = 'Epoch: [{}/{}]'.format(epoch, self.epochs)
        for batch_idx, data in enumerate(metric_logger.log_every(
            self.data_loader, 20, header)):
            
            # global training iteration
            it = len(self.data_loader) * (epoch - 1) + batch_idx
            # move images to gpu and perform augmentation to generate a image list
            images = self.augmentation(data['img'].cuda(non_blocking=True))

            # update weight decay and learning rate according to their schedule
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_scheduler[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_scheduler[it]

            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = self.student(images)
                loss = self.dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            self.optimizer.zero_grad()
            param_norms = None
            if self.fp16_scaler is None:
                loss.backward()
                if self.clip_grad:
                    param_norms = ddp_util_module.clip_gradients(self.student, self.clip_grad)
                ddp_util_module.cancel_gradients_last_layer(epoch, self.student,
                                                self.freeze_last_layer)
                self.optimizer.step()
            else:
                self.fp16_scaler.scale(loss).backward()
                if self.clip_grad:
                    self.fp16_scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = ddp_util_module.clip_gradients(self.student, self.clip_grad)
                ddp_util_module.cancel_gradients_last_layer(epoch, self.student,
                                                self.freeze_last_layer)
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = self.momentum_scheduler[it]  # momentum parameter
                for param_q, param_k in zip(self.student.module.parameters(), 
                                            self.teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # tensorboard logging setup
            self.writer.set_step(it)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=self.optimizer.param_groups[0]["weight_decay"])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        log = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        if self.val_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        Input:
            epoch <python int>: current training epoch.
        Output:
            log <python dict>: a log that contains information about validation
        """
        if self.distributed:
            self.val_data_loader.sampler.set_epoch(epoch)

        self.student.eval()
        self.teacher.eval()

        metric_logger = ddp_util_module.MetricLogger(delimiter="  ", 
                                writer=self.writer, logger=self.logger)
        header = 'Epoch: [{}/{}]'.format(epoch, self.epochs)
        epoch_embeddings = torch.tensor([])

        with torch.no_grad():
            for _, data in enumerate(metric_logger.log_every(
                self.data_loader, 1e6, header)):
                # move images to gpu and perform augmentation to generate a image list
                images = self.augmentation(data['img'].cuda(non_blocking=True))

                # only the 2 global views pass through the teacher
                teacher_output = self.teacher(images[:2])  
                student_output = self.student(images)
                loss = self.dino_loss(student_output, teacher_output, epoch)
                embeddings = torch_functional.normalize(
                    self.student.module.backbone(images[0]).detach().cpu())
            
                epoch_embeddings = torch.cat((epoch_embeddings, embeddings))
                torch.cuda.synchronize()
                metric_logger.update(val_loss=loss.item())

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            log = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            log['embedding_std'] = torch.mean(torch.std(epoch_embeddings, 0)).item()

        self.student.train()
        self.teacher.train()

        return log
    
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        Inputs:
            epoch <python int>: current epoch number
            save_best <python bool>: if True, rename the saved checkpoint to 
                'checkpoint_best.pth' and save to the desired location

        The self.learner, self.optimizer, and self.lr_scheduler are saved in the checkpoint
        The self.learner should have the net attribute, which is the backbone network
            function get_embedding_dimension(), get_drop_layer() and get_image_size() 
            should be implemented in the self.learner
        """
     
        # prepare the checkpoint dictionary for resuming training
        student_state_dict = self.student.module.state_dict() if hasattr(
            self.student, 'module') else self.student.state_dict()
        teacher_state_dict = self.teacher.module.state_dict() if hasattr(
            self.teacher, 'module') else self.teacher.state_dict()
        fp16_scaler_state = self.fp16_scaler.state_dict() if \
            self.fp16_scaler is not None else None

        checkpoint = {'epoch': epoch,
            'student': student_state_dict,
            'teacher': teacher_state_dict,
            'fp16_scaler': fp16_scaler_state,
            'optimizer': self.optimizer.state_dict(),
            'dino_loss': self.dino_loss.state_dict(),}
        # if self.lr_scheduler is not None:
        #     checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
    
        # save the checkpoint
        filename = str(self.save_dir / f'checkpoint-epoch{epoch}.pth')
        if epoch % self.save_period == 0:
            ddp_util_module.save_on_master(checkpoint, filename)
            self.logger.info(f"Saving checkpoint on epoch {epoch} ...")

        # save the best checkpoint
        if save_best:
            best_path = str(self.save_dir / 'checkpoint_best.pth')
            ddp_util_module.save_on_master(checkpoint, best_path)
            self.logger.info("Saving current best: checkpoint_best.pth ...")
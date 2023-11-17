from trainer_ssl import SSL_BaseTrainer
import logger_utils as module_log
import ddp_utils as ddp_util_module
import torch
import sys, math
import torch.nn.functional as torch_functional

# Class Inheritance of trainer_dino.py
# |--SSL_BaseTrainer (from trainer_ssl.py)
#      |--DINO_Trainer


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
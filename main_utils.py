import torch
import torch.nn.modules.utils as torch_utils
import numpy as np
import torchvision.transforms as transforms
import torchxrayvision.datasets as module_xrv_data
import platform
from pathlib import Path, PureWindowsPath
import json, os
import collections
import utils.customized_scheduler as module_scheduler
import torch.distributed as module_dist


def fix_random_seeds(seed=0):
    """
    Fix random seeds.
    Usage:
        main_ards_task.py
        main_ssl.py
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def is_dist_avail_and_initialized():
    if not module_dist.is_available():
        return False
    if not module_dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return module_dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return module_dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_transformation(image_size, transformation_code):
    """
    Obtain transformation for the data with transformation code.
    Inputs:
        image_size <python int>: the size of the image
        transformation_code <python str>: the code for the transformation
    Output:
        <python function>: the transformation function
    Usage:
        main_ards_task.py
        >>> module_utils.get_transformation(224, 'imagenet')
    """
    if transformation_code == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif transformation_code == 'default':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    elif transformation_code == 'xrv':
        transform = transforms.Compose([
            module_xrv_data.XRayCenterCrop(),
            module_xrv_data.XRayResizer(image_size)
        ])
    return transform

def read_json(fname):
    """Used in parse_config.py for loading the config file."""
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=collections.OrderedDict)


def write_json(content, fname):
    """Used in parse_config.py for saving the config file."""
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use):
    """
    Setup GPU device if available. 
    Get gpu device indices which are used for DataParallel.
    Usage:
        main_ssl.py
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def resume_from_checkpoint(ckp_path, logger, restore_variables=None, **kwargs):
    """
    Re-start the training from checkpoint. This action needs to be performed 
        before the Trainer initialization, therefore in the main_utils.py but 
        not in the Trainer class.

    Inputs:
        ckp_path: path to the checkpoint file
        restore_variables: a dictionary of variables to be loaded from the checkpoint
        kwargs: a dictionary of objects to be loaded from the checkpoint
    Usage:
        main_ssl.py
            resume_from_checkpoint(ckp_path, 
                restore_variables={"epoch": 0},
                learner=learner,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )
    For loading possible DDP models, the following code can be used:

    model_state_dict = torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                     checkpoint['model_dict'],prefix='module.')
    """
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError("No checkpoint found at '{}'".format(ckp_path))
    
    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # reload checkpoint into the desired modules
    ckp_name = Path(ckp_path).name
    for key, object_to_load in kwargs.items():
        if key in checkpoint and object_to_load is not None:
            checkpoint_refined = torch_utils.consume_prefix_in_state_dict_if_present(
                checkpoint[key], prefix='module.')
            try:
                msg = object_to_load.load_state_dict(checkpoint_refined, strict=False)
                logger.info("=> loaded {} from checkpoint '{}' with msg {}".format(
                    key, ckp_name, msg))
            except TypeError:
                try:
                    msg = object_to_load.load_state_dict(checkpoint_refined)
                    logger.info("=> loaded {} from checkpoint '{}'".format(
                        key, ckp_name))
                except ValueError:
                    logger.info("=> failed to load {} from checkpoint '{}'".format(
                        key, ckp_name))
        else:
            logger.info("=> failed to load {} from checkpoint '{}'".format(
                key, ckp_name))

    # reload variable important for the run                         
    if restore_variables is not None:
        for var_name in restore_variables:
            if var_name in checkpoint:
                restore_variables[var_name] = checkpoint[var_name]
                logger.info("=> loaded {} from checkpoint '{}'".format(
                    var_name, ckp_name))


def build_optimizer_and_scheduler(config, trainable_params):
    """
    Build optimizer and scheduler from config file
    Inputs:
        config: config file
        trainable_params: trainable parameters
    Useage:
        main_ssl.py
    """
    if hasattr(torch.optim, config['optimizer']['type']):
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    else:
        raise ValueError('Invalid optimizer type: {}'.format(config['optimizer']['type']))
    
    schedular_name = config['lr_scheduler']['type']
    if hasattr(torch.optim.lr_scheduler, schedular_name):
        lr_scheduler = config.init_obj(
            'lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        # import the customized scheduler as module_scheduler
        if schedular_name == 'LinearWarmupCosineAnnealingLR':
            scheduler_args = dict(config['lr_scheduler']['args'])
            scheduler_args['max_epochs'] = config['trainer']['epochs']
            scheduler_args['warmup_start_lr'] = 0.01*float(config['optimizer']['args']['lr'])
            scheduler_args['lr_eta_min'] = 1e-5
            lr_scheduler = getattr(module_scheduler, schedular_name)(optimizer, **scheduler_args)
        else: 
            raise NotImplementedError('Invalid lr_scheduler type: {}'.format(schedular_name))

    return optimizer, lr_scheduler      


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0):
    """
    The cosine_scheduler() function generates a learning rate schedule using the cosine 
    annealing technique for a given number of epochs and iterations per epoch. The 
    learning rate starts from base_value and gradually reduces to final_value 
    following the cosine curve. If a warmup_epochs value is provided, the function 
    linearly increases the learning rate from start_warmup_value to base_value for 
    the first warmup_epochs epochs. The output of the function is a 1D numpy array 
    containing the learning rate schedule for all epochs and iterations per epoch.

    Inputs:
        base_value <python float>: The base learning rate value.
        final_value <python float>: The final learning rate value.
        epochs: The total number of epochs to train.
        niter_per_ep: The number of iterations per epoch.
        warmup_epochs (optional, default: 0): The number of epochs to apply linear warmup.
        start_warmup_value (optional, default: 0): The starting learning rate value for linear warmup.
    
    The function first checks if there is any warmup epochs to be applied, and if so, 
    it generates a linear warmup schedule from start_warmup_value to base_value. Then, 
    it creates an array iters containing the total number of iterations, excluding the 
    warmup iterations. The cosine annealing schedule is calculated for each iteration 
    using the formula 
        final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters))). 
    Finally, the warmup schedule is concatenated with the cosine annealing schedule, 
    and the function returns the learning rate schedule as a numpy array. The function 
    also performs an assertion check to ensure that the length of the learning rate 
    schedule is equal to the total number of iterations.
    
    Usage:
        main_dino_ddp.py
    """

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_run_id(config_dict):
    """
    Get the run id from the config file
    Usage:
        main_ssl.py (but being applied in parse_config.py)
    """
    run_id = None
    arch_infor = config_dict['arch']['args']
    dataset_infor = config_dict['data_loader']['args']
    optimizer_infor = config_dict['optimizer']['args']
    lr_scheduler_infor = config_dict['lr_scheduler']['args'] if 'lr_scheduler' in config_dict else None

    if config_dict['arch']['type'] == "BYOL":

        run_id = f"{arch_infor['backbone'].lower()}_pretrained_{arch_infor['pretrained']}" + \
                f"_decay_{arch_infor['moving_average_decay']}" + \
                f"_{config_dict['data_loader']['type'].lower()}_{dataset_infor['batch_size']}" + \
                f"_optim_{config_dict['optimizer']['type'].lower()}_lr_{optimizer_infor['lr']}"
        if "weight_decay" in optimizer_infor:
            run_id += f"_wd_{optimizer_infor['weight_decay']}"
        if lr_scheduler_infor is not None:
            run_id += f"_scheduler_{config_dict['lr_scheduler']['type'].lower()}" 
            if config_dict['lr_scheduler']['type'] == "LinearWarmupCosineAnnealingLR":
                run_id += f"_warmup_{lr_scheduler_infor['warmup_epochs']}"

    if config_dict['arch']['type'] == "DINO":

        run_id = f"{arch_infor['model_name'].lower()}_patch_{arch_infor['patch_size']}" + \
                f"_teatemp_{config_dict['loss']['args']['teacher_temp']}" + \
                f"_{config_dict['data_loader']['type'].lower()}_{dataset_infor['batch_size']}" + \
                f"_{dataset_infor['image_size']}" + f"_localcrop_{config_dict['augmentation']['args']['local_crops_number']}"\
                f"_optim_{config_dict['optimizer']['type'].lower()}_lr_{config_dict['lr_scheduler']['args']['base_value']}" +\
                f"_warm_{config_dict['lr_scheduler']['args']['warmup_epochs']}_{config_dict['lr_scheduler']['args']['start_warmup_value']}"

        if "wd_scheduler" in config_dict:
            run_id += f"_wd_{config_dict['wd_scheduler']['args']['base_value']}"
        if "momentum_scheduler" in config_dict:
            run_id += f"_motum_{config_dict['momentum_scheduler']['args']['base_value']}"
        
        run_id += f"_use_fp16_{config_dict['trainer']['use_fp16']}"

    elif config_dict['arch']['type'].startswith('TRAM'):
        encoder_infor = config_dict['encoder']
        if "torchxrayvision" in encoder_infor['type']:
            encoder_type = config_dict['encoder']['args']["weights"]
            run_id = f"XRV-{encoder_type}"
        else:
            run_id = f"{encoder_infor['type'].split('.')[-1].lower()}"

        if config_dict['trainer']['test_monitor']:
            run_id += f"_test_monitored"

        run_id += f"_freeze_{encoder_infor['args']['freeze']}"
        if not encoder_infor['args']['freeze']:
            lr_infor = f"_lre_{encoder_infor['args']['lr_encoder']}_lr_{encoder_infor['args']['lr']}"
        else:
            try:
                lr_infor = f"_lr_{optimizer_infor['lr']}"
            except:
                lr_infor = f"_lre_{encoder_infor['args']['lr_encoder']}_lr_{encoder_infor['args']['lr']}"
        run_id += f"_img_{dataset_infor['image_size']}_enc_{dataset_infor['uncertainty_encoding']}"
        run_id += f"_func_{dataset_infor['use_uncertainty_func']}"
        if "uncertainty_threshold" in dataset_infor:
            run_id += f"_thred_{dataset_infor['uncertainty_threshold']}"
        run_id += f"_bs_{dataset_infor['batch_size']}"\
                  f"_baselys_{arch_infor['base_predictor_layer']}" + \
                  f"_adlys_{arch_infor['add_predictor_layer']}" + \
                  f"_adhid_{arch_infor['add_predictor_hidden_dim']}" + \
                  f"_lf_{config_dict['loss']['args']['additon_loss_factor']}" + \
                  f"_optim_{config_dict['optimizer']['type'].lower()}"
        
        for key, value in config_dict['optimizer']['args'].items():
            if key == "lr":
                continue
            elif key == "weight_decay":
                run_id += f"_wd_{value}"
            else:
                run_id += f"_{key}_{value}"

        run_id += lr_infor

    elif config_dict['arch']['type'] == "Linear_Eval" or config_dict['arch']['type'] == "Vits_Linear_Eval":
        encoder_infor = config_dict['encoder']
        if "torchxrayvision" in encoder_infor['type']:
            encoder_type = config_dict['encoder']['args']["weights"]
            run_id = f"XRV-{encoder_type}"
        else:
            run_id = f"{encoder_infor['type'].split('.')[-1].lower()}"
        
        run_id += f"_freeze_{encoder_infor['args']['freeze']}"

        if encoder_infor['args']['weights'] == 'imagenet' or encoder_infor['args']['weights'] == 'random':
            run_id += f"_{encoder_infor['args']['weights']}"

        if not encoder_infor['args']['freeze']:
            lr_infor = f"_lre_{encoder_infor['args']['lr_encoder']}_lr_{encoder_infor['args']['lr']}"
        else:
            lr_infor = f"_lr_{optimizer_infor['lr']}"
        run_id += f"_img_{dataset_infor['image_size']}_bs_{dataset_infor['batch_size']}"\
                  f"_optim_{config_dict['optimizer']['type'].lower()}"
        
        for key, value in config_dict['optimizer']['args'].items():
            if key == "lr":
                continue
            elif key == "weight_decay":
                run_id += f"_wd_{value}"
            else:
                run_id += f"_{key}_{value}"
                
        run_id += lr_infor

        if 'task_name' in config_dict:
            # When doing linear evaluation, the task name can be set to the name of the pre-trained model
            # or a name given by the user
            if config_dict['task_name'] == 'weights_dependent':
                run_id = Path(config_dict['encoder']['args']["weights"]).parent.name
            elif config_dict['task_name'] == 'weights_dependent_epoch':
                run_id = Path(config_dict['encoder']['args']["weights"]).parent.parent.name + '_' + \
                    Path(config_dict['encoder']['args']["weights"]).name.split('.')[0]
            else:
                run_id = config_dict['task_name']

    elif config_dict['arch']['type'] == "BYOL_TRAM":
        run_id = arch_infor['backbone'].lower() + f"_pretrain_{int(arch_infor['pretrained'])}"
        if encoder_infor['args']['weights'] == 'imagenet' or encoder_infor['args']['weights'] == 'random':
            run_id += f"_{encoder_infor['args']['weights']}"
            
        run_id += f"_pi_emb_{arch_infor['pi_encode_embed_dim']}_hid_{arch_infor['pi_encode_hid_dim']}" + \
            f"_proj_{arch_infor['pi_encode_project_dim']}_decay_{arch_infor['moving_average_decay']}"
            # "_byol_proj_{arch_infor['projection_size']}_hid_{arch_infor['projection_hidden_size']}"
        run_id += f"_{config_dict['data_loader']['type'].lower()}_bs_{dataset_infor['batch_size']}"\
                  f"_encode_{dataset_infor['encoding']}"
        if "loss_factor" in arch_infor:
            run_id += f"_lf_{arch_infor['loss_factor']}"

        if "lr" in config_dict:
            run_id += f"_optim_{config_dict['optimizer']['type'].lower()}"
            run_id += f"_lr_byol_{config_dict['lr']['byol']}_tram_{config_dict['lr']['tram_network']}"

            for key, value in config_dict['optimizer']['args'].items():
                if key == "weight_decay":
                    run_id += f"_wd_{value}"
                elif key == "momentum":
                    run_id += f"_mom_{value}"
                else:
                    run_id += f"_{key}_{value}"

            if lr_scheduler_infor is not None:
                run_id += f"_scduler_{config_dict['lr_scheduler']['type'].lower()}" 
                if config_dict['lr_scheduler']['type'] == "CosineAnnealingLR":
                    run_id += f"_etamin_{lr_scheduler_infor['eta_min']}"
                if config_dict['lr_scheduler']['type'] == "LinearWarmupCosineAnnealingLR":
                    run_id += f"_warmup_{lr_scheduler_infor['warmup_epochs']}"
        else:
            run_id += f"_optim_{config_dict['optimizer']['type'].lower()}"
            for key, value in config_dict['optimizer']['args'].items():
                if key == "weight_decay":
                    run_id += f"_wd_{value}"
                elif key == "momentum":
                    run_id += f"_mom_{value}"
                else:
                    run_id += f"_{key}_{value}"
            run_id += f"_optim_{config_dict['optimizer_tram']['type'].lower()}"
            for key, value in config_dict['optimizer_tram']['args'].items():
                if key == "weight_decay":
                    run_id += f"_wd_{value}"
                elif key == "momentum":
                    run_id += f"_mom_{value}"
                else:
                    run_id += f"_{key}_{value}"
            if lr_scheduler_infor is not None:
                lr_sche_type = config_dict['lr_scheduler']['type']
                if lr_sche_type == "CosineAnnealingLR":
                    run_id += f"_scduler_cosanl"
                    run_id += f"_etamin_{lr_scheduler_infor['eta_min']}"
                elif lr_sche_type == "LinearWarmupCosineAnnealingLR":
                    run_id += f"_scduler_lwcosanl"
                    run_id += f"_warmup_{lr_scheduler_infor['warmup_epochs']}" 
                else:
                    run_id += f"_scduler_{config_dict['lr_scheduler']['type'].lower()}"  

            lr_scheduler_infor_ = config_dict['lr_scheduler_tram']['args'] \
                if 'lr_scheduler_tram' in config_dict else None
            if lr_scheduler_infor_ is not None:
                lr_sche_type_ = config_dict['lr_scheduler_tram']['type']
                if lr_sche_type_ == "CosineAnnealingLR":
                    run_id += f"_scduler_cosanl"
                    run_id += f"_etamin_{lr_scheduler_infor_['eta_min']}"
                elif lr_sche_type_ == "LinearWarmupCosineAnnealingLR":
                    run_id += f"_scduler_lwcosanl"
                    run_id += f"_warmup_{lr_scheduler_infor_['warmup_epochs']}" 
                else:
                    run_id += f"_scduler_{config_dict['lr_scheduler']['type'].lower()}"  
    return run_id


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
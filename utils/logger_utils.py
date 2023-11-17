import importlib
from datetime import datetime
import logging
import logging.config
from pathlib import Path
from main_utils import read_json, is_main_process


class NoOp:
    """No operation class for handling logger when not in the main process."""
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass
        return no_op
    

class TensorboardWriter():
    """Tensorboard writer for logging training and validation metrics."""
    def __init__(self, log_dir, logger, enabled):

        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(
                        module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured " \
                    "to use, but currently not installed on this machine. Please install" \
                    " TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the" \
                    "option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional 
            information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined 
            # in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(
                    self.selected_module, name))
            return attr

def get_tensorboard_writer(log_dir, logger, enabled, distributed=False):
    """Get tensorboard writer.
    Inputs:
        log_dir <python str>: path to the directory where tensorboard logs are saved
        logger <logging.logger>: logger for logging information
        enabled <python bool>: whether to use tensorboard for visualization
        distributed <python bool>: whether distributed training is used.
            if not used or rank is 0, create the writer;
            otherwise, create a null writer, which is useful in DDP training.
    """
    if not distributed or is_main_process():
        return TensorboardWriter(log_dir, logger, enabled)
    else:
        return NoOp()

def get_logger(name, verbosity=2, rank=0):
    """
    Get logger with specified name and verbosity level
    Inputs:
        name <python str>: name of the logger
        verbosity <python int>: verbosity level of the logger
        rank <python int>: rank of the current process. If rank is 0, create the logger; 
            otherwise, create a null logger, which is useful in DDP training. 
    """
    if rank == 0:
        log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, log_levels.keys())
        assert verbosity in log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(log_levels[verbosity])
        return logger
    else:
        return NoOp()

def setup_logging_config(save_dir, log_config='logger_config.json', 
                         default_level=logging.INFO):
    """
    Setup logging configuration based on the configuration file.
    Used in parse_config.py to 
    """
    if is_main_process():
        log_config = Path(log_config)
        if log_config.is_file():
            config = read_json(log_config)
            # modify logging paths based on run config
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(save_dir / handler['filename'])

            logging.config.dictConfig(config)
        else:
            print("Warning: logging configuration file is not found in {}.".format(
                log_config))
            logging.basicConfig(level=default_level)
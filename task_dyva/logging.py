from abc import ABCMeta, abstractmethod

import neptune.new as neptune
from torch.utils.tensorboard import SummaryWriter


class Logger(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def log_metrics(self, metrics, name, epoch, epoch_end,
                    iteration, anneal_param):
        pass

    @abstractmethod
    def log_sample_output(self, fig, epoch):
        pass


class NeptuneLogger(Logger):
    # This is basically a thin wrapper for Neptune's logger

    def __init__(self, logger_info, neptune_project, expt_name,
                 params, tags):
        if logger_info is not None:
            # Reload existing logger
            expt_id = logger_info['id']
            self.logger = neptune.init(project=neptune_project, run=expt_id)
            self.is_new = False
        else:
            self.logger = neptune.init(project=neptune_project, name=expt_name)
            self.logger['parameters'] = params
            self.logger['sys/tags'].add(tags)
            self.info = self.logger.fetch()['sys']
            self.is_new = True

    def log_metrics(self, metrics, name, epoch, epoch_end=False,
                    iteration=None, anneal_param=None):
        for key, val in metrics.items():
            self.logger[f'{name}_{key}'].log(step=epoch, value=val)
        if epoch_end:
            self.logger['iteration'].log(step=epoch, value=iteration)
            self.logger['anneal_param'].log(step=epoch, value=anneal_param)

    def log_sample_output(self, fig, epoch):
        assign_key = f'model_outputs/epoch{epoch}'
        self.logger[assign_key].log(fig)


class TensorBoardLogger(Logger):
    # This is a thin wrapper for SummaryWriter (PyTorch class 
    # for writing info to TensorBoard)

    def __init__(self, expt_name, log_save_dir):
        self.logger = SummaryWriter(f'{log_save_dir}/{expt_name}')

    def log_metrics(self, metrics, name, epoch, epoch_end=False,
                    iteration=None, anneal_param=None):
        for key, val in metrics.items():
            self.logger.add_scalar(f'{name}_{key}', val, epoch)
        if epoch_end:
            self.logger.add_scalar('iteration', iteration, epoch)
            self.logger.add_scalar('anneal_param', anneal_param, epoch)

    def log_sample_output(self, fig, epoch):
        assign_key = f'model_outputs/epoch{epoch}'
        self.logger.add_figure(assign_key, fig, global_step=epoch)

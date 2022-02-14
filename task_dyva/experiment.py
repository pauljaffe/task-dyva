"""Defines classes which serve as a high-level interface 
for model training.
"""
import pkgutil
import os
import random
import pickle
import glob
import re
import copy

import torch
from torch.optim import Adadelta, Adam
from torch import nn
import neptune.new as neptune
import yaml
import numpy as np
import matplotlib.pyplot as plt

from .taskdataset import EbbFlowDataset, EbbFlowStats
from .utils import custom_collate, median_absolute_dev, ConfigMixin
from .transforms import AddStimulusNoise, FilterOutliers
from . import objectives
from . import models


class Experiment(nn.Module, 
                 ConfigMixin):
    """Defines methods for customizing model training.

    By default, model training metrics will be logged using Neptune
    (recommended; see https://neptune.ai/), but this can be disabled 
    by setting 'do_logging' in kwargs to False. If logging is enabled,
    neptune_proj_name should be supplied as an argument,
    and an API token should be established for authentication.

    See https://docs.neptune.ai/getting-started/installation 
    for detailed instructions. 

    Args
    ----
    base_dir (str): Directory where model training info 
        will be saved (checkpoints, logger info, etc.).
    raw_data_dir (str): Directory containing the raw/preprocessed
        data to prep for model training. 
    raw_fn (str): File name of the raw/preprocessed data to 
        load and process for model training. Must be located within
        raw_data_dir. Example: 'user1096.pickle'. 
    expt_name (str): Name of the experiment as it will be logged 
        in Neptune.
    expt_tags (list, optional): Tags to associate with the model training run
        in Neptune. 
    config (str, optional): If None, the default config in 
        config/model_config.yaml is used. If a filename is supplied, 
        the file should be located in base_dir. 
    device (str, optional): The device that will be used to train 
        the model (either 'cpu' or the default, 'cuda:0').
    processed_dir (str, optional): Directory in which to save the 
        processed data. If None (default), processed data will be saved in
        raw_data_dir. Another sensible choice would be base_dir,
        which contains the other experiment info.
    pre_transform (list of callables, optional): Transformations which 
        are applied before the processed data is saved. Some pre_transforms
        may also be defined in the config file; see _get_transforms method.
    transform (list of callables, optional): Transformations which are applied
        'online' on each iteration of the training loop. Some transforms
        may also be defined in the config file; see _get_transforms method. 
    pre_transform_params (list of dicts, optional): List of parameters to 
        pass to the pre_transforms. This should be a list of dictionaries
        with the same length as pre_transform.
    transform_params (list of dicts, optional): List of parameters to 
        pass to the transforms. This should be a list of dictionaries
        with the same length as transform.
    load_epoch (int, optional): Epoch to load parameters from when resuming
        model training. If None (default) and training is being resumed, the 
        most recently saved checkpoint epoch will be used. This arg
        is ignored for new training runs. 
    neptune_proj_name (str, optional): Name of the project as defined by
        the Neptune logger. 
    **kwargs (optional): Additional key: value mappings which can be passed
        to override the parameters defined in the config file. 
        See ConfigMixin.
    """

    def __init__(self, base_dir, raw_data_dir, raw_fn, expt_name, 
                 expt_tags=None, config=None, device='cuda:0', 
                 processed_dir=None, pre_transform=None, transform=None, 
                 pre_transform_params=None, transform_params=None, 
                 load_epoch=None, neptune_proj_name=None, **kwargs):
        super(Experiment, self).__init__()
        self.base_dir = base_dir
        self.expt_name = expt_name
        self.neptune_proj_name = neptune_proj_name
        self.expt_tags = [] if expt_tags is None else expt_tags
        self.device = device
        self.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        self._load_config(config, **kwargs)
        self.data_path = os.path.join(raw_data_dir, raw_fn)
        self.processed_dir = raw_data_dir if processed_dir is None \
            else processed_dir
        self.pre_transform = [] if pre_transform is None else pre_transform
        self.transform = [] if transform is None else transform
        self.pre_transform_params = [] if pre_transform_params is None \
            else pre_transform_params
        self.transform_params = [] if transform_params is None \
            else transform_params

        # enforce reproducibility
        rand_seed = self.config_params['training_params']['rand_seed']
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        self.rng = np.random.default_rng(rand_seed)

        self.objective = getattr(
            objectives, self.config_params['training_params']['objective'])
        self.clip_grads = self.config_params['training_params']['clip_grads']
        self.clip_val = self.config_params['training_params']['clip_val']
        self.keep_every = self.config_params['data_params']['keep_every']
        self.num_epochs = self.config_params['training_params']['num_epochs']
        self.update_every = self.config_params['training_params'][
            'temp_update_every']

        self._check_resume_training(load_epoch=load_epoch)
        self._prep_data()
        self._build_model()
        self._build_optimizer()
        if self.do_early_stopping:
            self._setup_early_stopping()

    @property
    def _max_anneal(self):
        return torch.tensor([1.0], device=self.device, requires_grad=False)

    def run_training(self):
        """Run the model training loop."""
        train_loader = self._get_data_loader(self.train_dataset)
        val_loader = self._get_data_loader(self.val_dataset)
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train()
            train_NLL, train_loss = self._batch_train(train_loader, 'train')

            self.eval()
            with torch.no_grad():
                train_to_log = {'loss': train_loss, 'NLL': train_NLL}
                self._log_metrics(train_to_log, 'train', epoch)

                if epoch % self.keep_every == 0:
                    val_NLL, val_loss = self._batch_train(val_loader, 'val')
                    val_to_log = {'loss': val_loss, 'NLL': val_NLL}
                    self._log_metrics(val_to_log, 'val', epoch)
                    self._save_checkpoint(epoch)
                    val_metrics = self.get_behavior_metrics(self.val_dataset, 
                                                            epoch, 'val')
                    self._log_metrics(val_metrics.summary_stats, 'val', 
                                      epoch, epoch_end=True)

                    if self.do_early_stopping:
                        self.early_stopping(val_metrics.summary_stats, epoch)
                        if self.early_stopping.early_stop:
                            print(f'Early stopping at epoch {epoch}')
                            break

    def _get_data_loader(self, dataset, shuffle=True):
        n_workers = self.config_params['training_params'].get('n_workers', 0)
        batch_size = self.config_params['training_params']['batch_size']
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            n_workers = int(n_workers * n_gpus)

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=n_workers,
                                             pin_memory=True,
                                             collate_fn=custom_collate,
                                             shuffle=shuffle)
        return loader

    def _batch_train(self, loader, mode):
        NLL_tot = 0
        loss_tot = 0
        for batch in loader:
            loaded_batch = batch.batch.to(self.device)
            n_time = loaded_batch.shape[0]

            if mode == 'train':
                for param in self.parameters():
                    param.grad = None

                if self.iteration % self.update_every == 0:
                    self._update_anneal_param()
                this_NLL, this_loss = self.objective(self.model, loaded_batch,
                                                     self.anneal_param)
                this_loss.backward()

                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 
                                                   self.clip_val)
                self.optimizer.step()
                self.iteration += 1

            elif mode == 'val':
                this_NLL, this_loss = self.objective(
                    self.model, loaded_batch, self._max_anneal)

            loss_tot += this_loss.item() / n_time
            NLL_tot += this_NLL.item() / n_time

        loss_avg = loss_tot / len(loader)
        NLL_avg = NLL_tot / len(loader)
        return NLL_avg, loss_avg

    def _load_config(self, config, **kwargs):
        if config is None:
            # Load default config
            self.config_params = yaml.safe_load(
                pkgutil.get_data('task_dyva', 'config/model_config.yaml'))
        else:
            config_path = os.path.join(self.base_dir, config)
            with open(config_path, 'r') as fn:
                self.config_params = yaml.safe_load(fn)
        # Optionally update config parameters with call to ConfigMixin method
        self.update_params(**kwargs)

    def _check_resume_training(self, load_epoch=None):
        if self.do_logging:
            # Check for saved logger and load most recent checkpoint;
            # create new experiment otherwise. 
            try:
                self._reload_logger()
                self._load_logged_checkpoint(epoch=load_epoch)
            except FileNotFoundError:
                self._make_new_experiment()
        else:
            # If load_epoch is None, the most recent local checkpoint is
            # loaded. If no checkpoints are found, a new experiment is
            # created. 
            self._load_local_checkpoint(epoch=load_epoch)

    def _reload_logger(self):
        project_name = self.neptune_proj_name
        with open(os.path.join(
                self.base_dir, 'logger_info.pickle'), 'rb') as handle:
            logger_info = pickle.load(handle)
        expt_id = logger_info['id']
        self.logger = neptune.init(project=project_name, run=expt_id)

    def _make_new_experiment(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = 0
        self.iteration = 0
        self.checkpoint = None
        if self.do_logging:
            project_name = self.neptune_proj_name
            self.logger = neptune.init(project=project_name, 
                                       name=self.expt_name)
            self.logger['parameters'] = self.config_params
            self.logger['sys/tags'].add(self.expt_tags)
            logger_info = self.logger.fetch()['sys']
            logger_path = os.path.join(self.base_dir, 'logger_info.pickle')
            with open(logger_path, 'xb') as handle:
                pickle.dump(logger_info, handle, protocol=4)

    def _load_local_checkpoint(self, epoch=None):
        if epoch is None:
            # Find epoch of most recent local checkpoint; if none exists,
            # create a new experiment.
            checkpoint_fns = glob.glob(self.checkpoint_dir 
                                       + 'checkpoint_epoch*.pth')
            if len(checkpoint_fns) == 0:
                self._make_new_experiment()
                return
            else:
                checkpoint_fns.sort(key=lambda f: int(re.sub(r'\D', '', f)))
                self.checkpoint = torch.load(checkpoint_fns[-1], 
                                             map_location=self.device)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, 
                                           f'checkpoint_epoch{epoch}.pth')
            self.checkpoint = torch.load(checkpoint_path, 
                                         map_location=self.device)
        try:
            self.iteration = self.checkpoint['iteration']
            self.start_epoch = self.checkpoint['epoch'] + 1
        except:
            self.iteration = 0
            self.start_epoch = 0

    def _load_logged_checkpoint(self, epoch=None):
        if epoch is None:
            # Find epoch of most recent checkpoint
            epoch = int(self.logger['checkpoint_epoch'].fetch_last())
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        dl_path = os.path.join(self.checkpoint_dir, 
                               f'checkpoint_epoch{epoch}.pth') 
        self.logger[f'model_checkpoints/epoch{epoch}'].download(dl_path)
        self._load_local_checkpoint(epoch=epoch)
        # Update iteration and start_epoch to be one plus the values that
        # were last logged to Neptune rather than those that were
        # last saved into a checkpoint (values logged to Neptune
        # must be strictly increasing). 
        self.iteration = int(self.logger['iteration'].fetch_last())
        logged_epochs = self.logger['train_loss'].fetch_values().index
        self.start_epoch = logged_epochs[-1] + 1

    def _log_metrics(self, metrics, name, epoch, epoch_end=False):
        if not self.do_logging:
            return

        for key, val in metrics.items():
            self.logger[f'{name}_{key}'].log(step=epoch, value=val)
        if epoch_end:
            self.logger['iteration'].log(step=epoch, value=self.iteration)
            self.logger['anneal_param'].log(step=epoch, 
                                            value=self.anneal_param)

    def _save_checkpoint(self, epoch):
        _ = self.plot_generated_sample(epoch, self.val_dataset)
        save_str = 'checkpoint_epoch{0}.pth'.format(epoch)
        save_path = os.path.join(self.checkpoint_dir, save_str)
        if self.do_early_stopping:
            save_counter = self.early_stopping.counter
            save_score = self.early_stopping.best_score
        else:
            save_counter = None
            save_score = None

        torch.save({'model_state': self.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'iteration': self.iteration,
                    'stop_counter': save_counter,
                    'stop_best_score': save_score},
                   save_path)
        if self.do_logging:
            # Also save checkpoint to Neptune
            self.logger['checkpoint_epoch'].log(epoch)
            self.logger[f'model_checkpoints/epoch{epoch}'].upload(save_path)

    def _prep_data(self):
        with open(self.data_path, 'rb') as handle:
            pre_processed = pickle.load(handle)
        filtered_data = filter_nth_play(
            pre_processed, 
            self.config_params['data_params']['nth_play_range'])
        outlier_params = _get_outlier_params(filtered_data, 
                                             self.config_params)

        pre_datasets = self._split_train_val_test(filtered_data)
        splits = ['train', 'val', 'test']
        if self.mode == 'training':
            datasets_to_prep = pre_datasets
            splits_to_prep = splits
            splits_to_set = ['train', 'val']
        elif self.mode == 'testing':
            datasets_to_prep = [pre_datasets[2]]
            splits_to_prep = [splits[2]]
            splits_to_set = [splits[2]]
        elif self.mode == 'val_only':
            datasets_to_prep = [pre_datasets[1]]
            splits_to_prep = [splits[1]]
            splits_to_set = [splits[1]]
        elif self.mode == 'full':
            datasets_to_prep = pre_datasets
            splits_to_prep = splits
            splits_to_set = splits

        for pre_data, split in zip(datasets_to_prep, splits_to_prep):
            this_params = self.config_params['data_params'].get(
                f'{split}_transform_kwargs')
            this_params['outlier_params'] = outlier_params
            trans, pre_trans, trans_params, pre_trans_params = \
                self._get_transforms(this_params)
            dataset = EbbFlowDataset(
                self.base_dir, this_params, pre_data, split, 
                pre_transform=pre_trans, transform=trans,
                pre_transform_params=pre_trans_params, 
                transform_params=trans_params,
                processed_dir=self.processed_dir)
            if split in splits_to_set:
                setattr(self, f'{split}_dataset', dataset)

    def _get_transforms(self, params):
        pre_transform = copy.deepcopy(self.pre_transform)
        transform = copy.deepcopy(self.transform)
        pre_transform_params = copy.deepcopy(self.pre_transform_params)
        transform_params = copy.deepcopy(self.transform_params)

        outlier_method = params['outlier_params'].get('method', None)
        if outlier_method in ['mad']:
            pre_transform.append(FilterOutliers)
            pre_transform_params.append(params)
        if len(pre_transform) == 0:
            pre_transform = None
            pre_transform_params = None

        noise_type = params.get('noise_type', None)
        if noise_type in ['indep', 'corr']:
            transform.append(AddStimulusNoise)
            transform_params.append(params)
        if len(transform) == 0: 
            transform = None
            transform_params = None
        return transform, pre_transform, \
            transform_params, pre_transform_params

    def _build_model(self):
        mtype = getattr(models,
                        self.config_params['model_params']['model_type'])
        self.model = mtype(self.config_params)
        self.to(self.device)

        if self.checkpoint is not None:
            self.load_state_dict(self.checkpoint['model_state'])
        self._update_anneal_param()

    def _build_optimizer(self):
        optim_alg = self.config_params['training_params']['optim_alg']
        lr = self.config_params['training_params']['LR']
        weight_decay = self.config_params['training_params']['weight_decay']
        do_amsgrad = self.config_params['training_params'].get(
            'do_amsgrad', True)
        if optim_alg == 'adadelta':
            self.optimizer = Adadelta(filter(lambda p: p.requires_grad, 
                                             self.parameters()),
                                      lr=lr, weight_decay=weight_decay)
        elif optim_alg == 'adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, 
                                         self.parameters()),
                                  lr=lr, weight_decay=weight_decay, 
                                  amsgrad=do_amsgrad)
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state'])

    def _setup_early_stopping(self):
        patience = self.config_params['training_params'].get(
            'stop_patience', 20)
        min_epoch = self.config_params['training_params'].get(
            'stop_min_epoch', 500)
        delta = self.config_params['training_params'].get(
            'stop_delta', 0)
        stop_metric = self.config_params['training_params'].get(
            'stop_metric', 'switch_con_avg')
        self.early_stopping = EarlyStopping(patience, min_epoch, delta,
                                            stop_metric)
        if self.checkpoint is not None:
            self.early_stopping.best_score = self.checkpoint.get(
                'stop_best_score', 0)
            self.early_stopping.counter = self.checkpoint.get(
                'stop_counter', 0)

    def plot_generated_sample(self, epoch, dataset, do_plot=False, 
                              sample_ind=None, stim_ylims=None,
                              resp_ylims=None):
        """Plot the model inputs and outputs for a single sample (game).

        Args
        ----
        epoch(int): Training epoch used to generate the sample. This is saved
            in the filename of the figure uploaded to Neptune if 
            self.do_logging=True; otherwise it does nothing. 
        dataset (EbbFlowDataset instance): Dataset from which the sample
            will be plotted (e.g. self.val_dataset).
        do_plot (Boolean, optional): Whether or not to plot the figure.
        sample_ind (int, optional): The index of the sample to plot. 
        stim_ylims (two-element list, optional): The y-axis limits for
            the stimuli.
        resp_ylims (two-element list, optional): The y-axis limits for
            the responses.

        Returns
        -------
        trial (EbbFlowGameData instance): Game data for the plotted sample. 
        """

        # Running a single sample (batch size = 1) through the model would
        # require changing the model implementation, so we pass through
        # two samples and throw away the second. 
        if sample_ind is None:
            n = len(dataset)
            gen_inds = self.rng.choice(n, 2)
        else:
            gen_inds = [sample_ind, 0]

        # Generate model outputs from dataset
        xu_sample = dataset.xu[:, gen_inds, :].to(self.device)
        gen_outputs = self.model.forward(xu_sample, generate_mode=True, 
                                         clamp=False)
        rate_out = gen_outputs[1].mean
        rate_np = rate_out.cpu().detach()[:, 0, :].squeeze().numpy()
        trial = dataset.get_processed_sample(gen_inds[0])
        trial.get_extra_stats(output_rates=rate_np)
        trial_fig, _ = trial.plot(rates=rate_np, stim_ylims=stim_ylims,
                                  resp_ylims=resp_ylims)

        if self.do_logging:
            assign_key = f'model_outputs/epoch{epoch}'
            self.logger[assign_key].log(trial_fig)
        if do_plot:
            plt.show()
        plt.close('all')

        return trial

    def forward(self, inputs):
        # Compute a forward pass of the model; see forward method
        # in model implementation.
        return self.model(inputs)

    def _update_anneal_param(self):
        st = self.config_params['training_params']['start_temp']
        cr = self.config_params['training_params']['cool_rate']
        anneal_param = torch.tensor([st + self.iteration / cr],
                                    device=self.device, requires_grad=False)
        self.anneal_param = torch.min(anneal_param, self._max_anneal)

    def get_behavior_metrics(self, dataset, epoch, label, save_local=False, 
                             load_local=True, analyze_latents=False,
                             get_model_outputs=True, stats_dir=None,
                             **kwargs):
        """Get behavioral stats for user and model from the supplied dataset.

        Args
        ----
        dataset (EbbFlowDataset instance): Dataset used to calculate stats.
        epoch (int): Epoch number saved into the file name, otherwise
            has no function.
        label (str): A label to save into the file name, otherwise
            has no function. 
        save_local (Boolean, optional): Whether or not to save a local copy
            of the resulting stats object.
        load_local (Boolean, optional): Whether or not to reload a saved
            stats object. 
        analyze_latents (Boolean, optional): Whether or not to analyze
            the latent state variables. 
        get_model_outputs (Boolean, optional): Whether or not to generate
            model outputs.
        stats_dir (str): Name of the directory within self.base_dir to
            save model stats.
        **kwargs (optional): Additional key, value pairs to pass to
            EbbFlowStats.

        Returns
        -------
        stats_obj (EbbFlowStats instance): Container with stats for the
            supplied dataset. 
        """

        if stats_dir is None:
            stats_dir = 'model_stats'
        stats_path = os.path.join(self.base_dir, stats_dir, save_str)
        if load_local:
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as path:
                    stats = pickle.load(path)
                return stats

        if analyze_latents:
            latents = []
        else:
            latents = None

        if get_model_outputs:
            loader = self._get_data_loader(dataset, shuffle=False)
            rates = []
            for batch_ind, batch in enumerate(loader):
                loaded_batch = batch.batch.to(self.device)
                outputs = self.model.forward(loaded_batch,
                                             generate_mode=True, clamp=False)
                rates.append(outputs[1].mean)
                if analyze_latents:
                    latents.append(outputs[3])

            rates = torch.cat(rates, 1)
            if analyze_latents:
                latents = torch.cat(latents, 1)
        else:
            rates = None

        stats_obj = EbbFlowStats(rates, dataset, latents=latents, **kwargs)
        stats = stats_obj.get_stats()

        if save_local:
            os.makedirs(os.path.join(self.base_dir, 'model_stats'), 
                        exist_ok=True)
            with open(stats_path, 'wb') as path:
                pickle.dump(stats_obj, path, protocol=4)

        return stats_obj

    def _split_train_val_test(self, data):
        if self.split_indices is not None:
            split_inds = self.split_indices
        else:
            save_str = os.path.join(self.base_dir, 'split_inds.pkl')
            if os.path.exists(save_str):
                with open(save_str, 'rb') as path:
                    split_inds = pickle.load(path)
            else:
                train_frac = self.config_params['training_params'][
                    'train_frac']
                val_frac = self.config_params['training_params']['val_frac']
                n_data = len(data['resp_time']) 
                n_train = int(np.round(n_data * train_frac))
                n_val = int(np.round(n_data * val_frac))
                indices = list(range(n_data))

                self.rng.shuffle(indices)
                train_inds = indices[:n_train]
                val_inds = indices[n_train:n_train + n_val]
                test_inds = indices[n_train + n_val:]

                split_inds = {'train_inds': train_inds, 
                              'val_inds': val_inds, 
                              'test_inds': test_inds}
                with open(save_str, 'xb') as path:
                    pickle.dump(split_inds, path, protocol=4)

        train, val, test = {}, {}, {}
        for key, vals in data.items():
            train[key] = vals[split_inds['train_inds']]
            val[key] = vals[split_inds['val_inds']] 
            test[key] = vals[split_inds['test_inds']]
        return train, val, test


class EarlyStopping:
    """Implements early stopping for model training. 

    Args
    ----
    patience (int): How many epochs to wait with improvement in stop_metric
        before stopping. 
    min_epoch (int): The minimum epoch that early stopping can occur. 
    delta (int): The score must exceed the best score by this amount
        to count as an improvement in score. 
    stop_metric (str): The metric to monitor for early stopping. 
        Currently, the only available option is 'switch_con_avg',
        which measures the discrepancy between user/model switch cost 
        and user/model congruency effect (see _get_stop_score 
        for implementation details). 
    """

    def __init__(self, patience, min_epoch, delta, stop_metric):
        self.patience = patience
        self.delta = delta
        self.stop_metric = stop_metric
        self.min_epoch = min_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metrics, epoch):
        if epoch < self.min_epoch:
            return
        score = self._get_stop_score(metrics)
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def _get_stop_score(self, metrics):
        if self.stop_metric == 'switch_con_avg':
            m_sc, u_sc = metrics['m_switch_cost'], metrics['u_switch_cost']
            m_ce, u_ce = metrics['m_con_effect'], metrics['u_con_effect']
            sc_diff = np.abs(m_sc - u_sc) / u_sc
            ce_diff = np.abs(m_ce - u_ce) / u_ce
            return sc_diff + ce_diff


def filter_nth_play(data, keep_range):
    """Filter gameplay data to only retain games for which the nth play 
    is within the range keep_range.

    Args
    ----
    data (dict): Preprocessed gameplay data. 
    keep_range (array like): An interval which defines the range of gameplays
        to keep. For example, to keep the first 100 games played by a 
        given user, set keep_range to [1, 100] (the index of the first 
        gameplay is 1). 

    Returns
    -------
    filtered_data (dict): Filtered gameplay data.
    """

    keep_inds = [ind for ind, nth in enumerate(data['nth_master'])
                 if keep_range[0] <= nth <= keep_range[1]]
    filtered_data = {key: vals[keep_inds] for key, vals in data.items()}
    return filtered_data


def _get_outlier_params(data, config_params):
    outlier_method = config_params['data_params'].get('outlier_method', None)
    rts = [rt for game in data['resp_time'] for rt in game]
    if outlier_method == 'mad':
        thresh = config_params['data_params']['outlier_thresh']
        mad, _ = median_absolute_dev(np.array(rts))
        median = np.median(np.array(rts))
        outlier_params = {'method': outlier_method, 'thresh': thresh, 
                          'mad': mad, 'median': median}
    else:
        outlier_params = {}
    return outlier_params

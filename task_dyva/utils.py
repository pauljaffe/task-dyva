import os
import itertools

import torch
import numpy as np
import pandas as pd
from scipy.stats import special_ortho_group, pearsonr
from sklearn.decomposition import PCA


class Constants(object):
    eta = 1e-6


class CustomBatch:
    def __init__(self, batch):
        self.batch = torch.stack(batch, dim=1)

    def pin_memory(self):
        self.batch = self.batch.pin_memory()
        return self


def custom_collate(batch):
    """Custom function for constructing batches on calls to DataLoader."""
    return CustomBatch(batch)


class ConfigMixin:
    """Mixin class which enables overriding parameters in the config file.
    This is useful when one wants to test out different sets of 
    model/data/training parameters without creating a new config file 
    for each set.
    """

    _model_params = {'latent_dim', 'init_rnn_dim', 'init_hidden_dim',
                     'w_dim', 'encoder_mlp_hidden_dim', 
                     'encoder_rnn_input_dim', 'encoder_rnn_hidden_dim',
                     'encoder_rnn_dropout', 'encoder_rnn_n_layers',
                     'encoder_combo_hidden_dim', 'decoder_hidden_dim',
                     'trans_dim', 'prior_dist', 'posterior_dist', 
                     'likelihood_dist', 'likelihood_scale_param',
                     'model_type', 'dynamics_matrix_mult', 
                     'dynamics_init_method'}

    _training_params = {'num_epochs', 'batch_size', 'train_frac', 'val_frac', 
                        'test_frac', 'optim_alg', 'LR', 'weight_decay', 
                        'clip_grads', 'clip_val', 'n_workers', 'rand_seed', 
                        'learn_prior', 'objective', 'do_amsgrad', 'start_temp',
                        'cool_rate', 'temp_update_every', 'stop_patience', 
                        'stop_min_epoch', 'stop_delta', 'stop_metric'}

    _data_params = {'input_dim', 'u_dim', 'nth_play_range', 'outlier_method', 
                    'outlier_thresh', 'keep_every'}

    _transform_splits = {'train', 'val', 'test'}

    _transform_params = {'step_size', 'duration', 'noise_type', 'noise_sd', 
                         'noise_corr_weight', 'start_times', 
                         'data_augmentation_type', 'data_aug_kernel_bandwidth',
                         'aug_rt_sd', 'aug_resample_frac', 'upscale_mult', 
                         'min_trials', 'post_resp_buffer', 'smoothing_type',
                         'kernel_sd', 'match_accuracy', 'rt_method'}

    _experiment_params = {'split_indices', 'processed_save_dir', 'mode',
                          'logger_type', 'do_early_stopping', 'params_to_load',
                          'neptune_proj_name', 'expt_tags', 'log_save_dir'}

    def update_params(self, **kwargs):
        # Update params from the config file
        for key, val in kwargs.items():
            if key in self._model_params:
                self.config_params['model_params'][key] = val
            elif key in self._training_params:
                self.config_params['training_params'][key] = val
            elif key in self._data_params:
                self.config_params['data_params'][key] = val
            elif key in self._transform_splits:
                # val must be a dictionary
                split_key = f'{key}_transform_kwargs'
                for tkey, tval in val.items():
                    if tkey in self._transform_params:
                        self.config_params['data_params'][split_key][
                            tkey] = tval
                    else: 
                        raise KeyError(f'Invalid config key: {tkey}')
            elif key in self._experiment_params:
                pass  # handled by _update_experiment_options
            else:
                raise KeyError(f'Invalid config key: {key}')
        # Also check for options not specified in the config file
        self._update_experiment_options(**kwargs)

    def _update_experiment_options(self, **kwargs):
        # Check for extra options not specified in the config file
        self.split_indices = kwargs.get('split_indices', None)
        self.processed_save_dir = kwargs.get('processed_save_dir', 
                                             'processed')
        self.mode = kwargs.get('mode', 'training')
        self.logger_type = kwargs.get('logger_type', 'tensorboard')
        self.do_early_stopping = kwargs.get('do_early_stopping', True)
        self.params_to_load = kwargs.get('params_to_load', None)
        self.neptune_proj_name = kwargs.get('neptune_proj_name', None)
        self.expt_tags = kwargs.get('expt_tags', [])
        self.log_save_dir = kwargs.get('log_save_dir', 'tensorboard')


def median_absolute_dev(data, median=None):
    """Determine the median absolute deviation from the median (MAD) 
    of the supplied dataset.

    Args
    ----
    data (array-like): The data to calculate the MAD of.
    median (float, optional): The median value used to calculate the MAD.
        If set to None, the median will be calculated from the supplied data.

    Returns
    -------
    mad (float): The calculated MAD. 
    devs (NumPy array): The absolute deviations from the median used to
        calculate the MAD. 
    """

    if median is None:
        median = np.median(data)
    devs = np.abs(data - median)
    mad = np.median(devs)
    return mad, devs


def _init_dynamics_mats(dim, n_mats, rand_seed, method):
    # Initialize the dynamics matrices
    rng = np.random.default_rng(rand_seed)

    if method == 'special_ortho':
        # Sample random rotation matrices
        dyn_mats = special_ortho_group.rvs(dim=dim, size=n_mats, 
                                           random_state=rand_seed)
    elif method == 'custom_rotation':
        # Sample random matrices with rotational dynamics
        # (note these are not true rotation matrices -- this method
        # works well in practice).
        all_mats = []
        for n in range(n_mats):
            block_R = np.zeros((dim, dim))
            theta = rng.random() * np.pi / 2
            cos, sine = np.cos(theta), np.sin(theta)
            block_R[:2, :2] = np.array([[cos, -sine], [sine, cos]])
            rand_mat = rng.normal(0, 1, (dim, dim))
            Q, _ = np.linalg.qr(rand_mat)
            all_mats.append(Q @ block_R @ Q.T)
        dyn_mats = np.stack(all_mats, axis=0)

    return torch.tensor(dyn_mats).type(torch.FloatTensor)


def z_pca(z, n_keep, whiten=False):
    # Transform the latent state vars to PCA space
    if torch.is_tensor(z):
        z_np = z.cpu().detach().numpy()
    else:
        z_np = z
    z_dim = z_np.shape[2]
    T = z_np.shape[0]
    N = z_np.shape[1]
    z_cat = np.reshape(z_np, (T * N, z_dim), order='F') # concatenate

    # Run PCA
    z_pca_obj = PCA(whiten=whiten).fit(z_cat)
    z_var_exp = z_pca_obj.explained_variance_ratio_
    z_transformed = z_pca_obj.transform(z_cat)
    z_reduced = np.reshape(
            z_transformed, (T, N, z_dim), order='F')[:, :, :n_keep]

    return z_reduced, z_var_exp, z_pca_obj


def save_figure(fig, save_dir, fn, save_svg=True, save_png=True):
    if save_svg:
        svg_path = os.path.join(save_dir, f'{fn}.svg')
        fig.savefig(svg_path, transparent=True, bbox_inches='tight')
    if save_png:
        png_path = os.path.join(save_dir, f'{fn}.png')
        fig.savefig(png_path, bbox_inches='tight')


def plot_scatter(group_stats, params, ax, line_ext):
    # Model vs. user scatter plotting utility
    metric = params['metric']
    u_key = f'u_{metric}'
    m_key = f'm_{metric}'
    u_vals = np.array(group_stats[u_key])
    m_vals = np.array(group_stats[m_key])    
    
    # Plot best fit line
    plot_x = np.array([min(u_vals) - line_ext, 
                       max(u_vals) + line_ext])
    m, b = np.polyfit(u_vals, m_vals, 1)
    ax.plot(plot_x, m * plot_x + b, 'r--', zorder=1, linewidth=0.5)

    # Plot unity line
    ax.plot(plot_x, plot_x, 'k-', zorder=1, linewidth=0.5)
    
    # Plot all individuals
    ax.scatter(u_vals, m_vals, s=0.2, marker='o', zorder=2, alpha=0.8)
    ax.set_xlabel(f"Participant {params['label']}")
    ax.set_ylabel(f"Model {params['label']}")
    ax.set_xlim(params['ax_lims'])
    ax.set_ylim(params['ax_lims'])

    # Stats
    r = pearsonr(u_vals, m_vals)
    u_mean = np.mean(u_vals)
    m_mean = np.mean(m_vals)
    u_sem = np.std(u_vals) / np.sqrt(len(u_vals))
    m_sem = np.std(m_vals) / np.sqrt(len(m_vals))
    print(f'{metric} corr. coeff.: {r[0]}, p = {r[1]}')
    print(f'{metric} slope: {m}; intercept: {b}') 
    print(f'Participant {metric} mean +/- s.e.m.: {u_mean} +/- {u_sem}')
    print(f'Model {metric} mean +/- s.e.m.: {m_mean} +/- {m_sem}')
    print('--------------------------------------------------------')


def expt_stats_to_df(metrics, expts, age_bins, stats):
    df = pd.DataFrame(
        columns=['user', 'age_bin', 'metric', 'value', 'model_or_user'])
    for expt, ab, stats in zip(expts, age_bins, stats):
        for key in metrics:
            u_key = 'u_{0}'.format(key)
            m_key = 'm_{0}'.format(key)
            u_val = stats.summary_stats[u_key]
            m_val = stats.summary_stats[m_key]
            new_u_row = {'user': expt, 'age_bin': ab, 'metric': key, 
                         'value': u_val, 'model_or_user': 'Participants'}
            new_m_row = {'user': expt, 'age_bin': ab, 'metric': key, 
                         'value': m_val, 'model_or_user': 'Models'}
            df = df.append(new_u_row, ignore_index=True)
            df = df.append(new_m_row, ignore_index=True) 
    return df


def get_stimulus_combos():
    # Returns tuples of all possible stimulus combinations for 
    # a single trial (move, point, task cue)
    combos = list(itertools.product([0, 1, 2, 3],
                                    [0, 1, 2, 3],
                                    [0, 1]))
    return combos

import torch
import numpy as np
from scipy.stats import special_ortho_group
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
                          'do_logging', 'do_early_stopping'}

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
        self.do_logging = kwargs.get('do_logging', True)
        self.do_early_stopping = kwargs.get('do_early_stopping', True)


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

    return z_reduced, z_var_exp

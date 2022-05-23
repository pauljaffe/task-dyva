"""Data transformations defined as classes with an implemented
__call__ method. These can be supplied as inputs to the TaskDataset class.
"""
import math
import pdb

import numpy as np
import torch

from .utils import median_absolute_dev as mabsdev


class _Trim():
    # Transform used to trim the duration of the continuous form of the
    # stimuli and responses from an EbbFlowGameData instance. By default,
    # this is called by EbbFlowDataset during processing. 

    def __call__(self, data):
        for key in data.continuous.keys():
            data.continuous[key] = \
                data.continuous[key][:data.num_steps_short_win, :]
        return data


class SmoothResponses():
    """Transform which smooths the continuous format of the user's responses
    from an EbbFlowGameData instance. The type of smoothing is specified
    by setting the 'smoothing_type' key in params to one of the following 
    options:

    'gaussian': Responses are smoothed with a Gaussian kernel.
        Currently this is the only available option. The width of the 
        smoothing kernel is reduced over the course of training from 
        'init_kernel_sd' to 'final_kernel_sd' over 'kernel_delta_epochs' epochs. 
    """
    # This could all be done in Numpy to be a bit more readable... 

    kernel_t_max = 3500

    def __init__(self, params):
        self.t = torch.arange(0, self.kernel_t_max, params['step_size'])
        self.init_sd = params['init_kernel_sd']
        self.final_sd = torch.Tensor([params['final_kernel_sd']])
        self.tau = params['kernel_delta_epochs']
        self.m = self.kernel_t_max / 2
        self.n_steps = int(np.rint(params['duration'] 
                                   / params['step_size']))

    def _update_sm_kernel(self, epoch):
        kernel_range = self.init_sd - self.final_sd
        sd_pre = torch.Tensor([self.init_sd - kernel_range * epoch / self.tau])
        sd = torch.max(sd_pre, self.final_sd)
        self.kernel = torch.exp(
            -0.5 * ((self.t - self.m) / sd) ** 2).unsqueeze(0).repeat(4, 1, 1)

    def __call__(self, data):
        """Smooth the responses of a single game. Also trim down
        to be the proper length.

        Args
        ----
        data (EbbFlowGameData instance): Game data object to be transformed.

        Returns
        -------
        data (EbbFlowGameData instance): Game data object after the 
            transformation is applied. 
        """

        sm_data = torch.clone(data)
        orig_resp = torch.transpose(sm_data[:, :4], 0, 1).unsqueeze(0)
        sm_resp = torch.nn.functional.conv1d(orig_resp, self.kernel, groups=4,
                                             padding='same')
        # Correct one timestep offset
        sm_data[1:, :4] = torch.transpose(sm_resp.squeeze(), 0, 1)[:-1, :]
        return sm_data[:self.n_steps, :]


class FilterOutliers():
    """Transform which removes outlier RTs. Outlier removal is specified
    with the following keys in params['outlier_params']:
    thresh (float): Threshold value used to exclude outliers. 
        See description below.
    method (str): The method used for outlier removal. Currently, the only 
        supported method is 'mad', which is based on the median absolute 
        deviation (MAD) of the RT distribution. The MAD is defined as the 
        median of the absolute deviations from the median RT. RTs which deviate 
        from the median by more than thresh times the MAD are excluded. If 
        method is set to 'mad', the keys 'mad' (the MAD of the dataset) and 
        'median' (the median RT of the dataset) must also be set. 

    Note: If any of the RTs in a given EbbFlowGameData instance are flagged
          as outliers, all trials from that instance are excluded from the
          dataset. 
    """

    supported_methods = ['mad']

    def __init__(self, params):
        self.method = params['outlier_params']['method']
        assert self.method in self.supported_methods, \
            'Outlier method not supported!'
        self.thresh = params['outlier_params']['thresh']
        if self.method == 'mad':
            self.mad = params['outlier_params']['mad']
            self.median = params['outlier_params']['median']

    def __call__(self, data):
        """Filter outlier RTs for a single game.

        Args
        ----
        data (EbbFlowGameData instance): Game data object to be transformed.

        Returns
        -------
        data (EbbFlowGameData instance): Game data object after the 
            transformation is applied. If any outliers are found,
            the is_valid attribute of data is set to False; this can be used
            to remove the game data from the processed dataset. 
        """

        rts = data.discrete['urt_ms']
        if len(rts) == 0:
            data.is_valid = False
        else:
            if self.method == 'mad':
                _, devs = mabsdev(np.array(rts), median=self.median)
                if len(devs[devs >= self.thresh * self.mad]) > 0:
                    data.is_valid = False  # entire game is excluded
        return data


class AddStimulusNoise():
    """Transform which adds noise to the continuous format of the stimuli from 
    an EbbFlowGameData instance. The parameters of the noise are defined using
    the following keys in params:

    noise_type (str): Must be either 'corr' for correlated noise or 
        'indep' for independent noise. If set to 'indep', independent
        Gaussian noise with SD set by noise_sd is added to each stimulus
        direction for the moving and pointing tasks as well as both 
        task cue stimuli. If set to 'corr', the noise added will be 
        correlated across the different stimulus channels, where the degree
        of correlation is controlled by noise_corr_weight. By default,
        the correlations within each stimulus modality will be stronger
        than the correlations between different stimulus modalities. 
        See _add_correlated_noise for implementation details. 
    noise_sd (float): The SD of the Gaussian noise to be added. 
    noise_corr_weight (float): Parameter which controls the strength of the
        noise correlations across the stimulus channels. This should be set
        to a value between 0 and 1. If set to 0, the noise across the stimulus 
        channels will be independent; if set to 1, the noise will be highly 
        correlated across channels. 
    """

    def __init__(self, params):
        self.params = params

    def __call__(self, data):
        """Add noise to the stimuli from a single game. 

        Args
        ----
        data (EbbFlowGameData instance): Game data object to be transformed.

        Returns
        -------
        data (EbbFlowGameData instance): Game data object after the 
            transformation is applied.
        """

        noisy_data = torch.clone(data)
        if self.params['noise_type'] == 'corr': 
            noisy_data = self._add_correlated_noise(noisy_data)
        elif self.params['noise_type'] == 'indep':
            noisy_data = self._add_indep_noise(noisy_data)
        return noisy_data

    def _add_indep_noise(self, data):
        std = self.params['noise_sd'] * torch.ones(data[:, 4:].shape)
        noise = torch.normal(mean=0.0, std=std)
        data[:, 4:] = (data[:, 4:] + noise).to(dtype=torch.float32)
        return data

    def _add_correlated_noise(self, data, latent_weights=[0.6, 0.2, 0.2]):
        noise_cw = self.params['noise_corr_weight']
        n_time = data.shape[0]
        latent_inds = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        n_dims = [4, 4, 2]
        start_inds = [4, 8, 12]
        latent_noise_sd = self.params['noise_sd'] * torch.ones((n_time, 3))
        latent_sources = torch.normal(mean=0.0, std=latent_noise_sd)
        for li, nd, si in zip(latent_inds, n_dims, start_inds):
            this_corr_noise = (
                math.sqrt(latent_weights[0]) * latent_sources[:, li[0]] 
                + math.sqrt(latent_weights[1]) * latent_sources[:, li[1]]
                + math.sqrt(latent_weights[2]) * latent_sources[:, li[2]])
            this_corr_noise = this_corr_noise.unsqueeze(1).repeat(1, nd)
            this_noise_sd = self.params['noise_sd'] * torch.ones((n_time, nd))
            this_indep_noise = torch.normal(mean=0.0, std=this_noise_sd)
            # The transformation below ensures that the resulting
            # noise has SD ~= self.params['noise_sd']
            this_noise = (math.sqrt(noise_cw) * this_corr_noise 
                          + math.sqrt(1 - noise_cw) * this_indep_noise)
            data[:, si:si + nd] = (
                data[:, si:si + nd] + this_noise).to(dtype=torch.float32)
        return data


class Compose():
    """Used to chain together a sequence of transformations.

    Args
    ----
    transforms (list of callables): The sequence of transformations to 
        be applied.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        """Apply the sequence of transformations specified in the transforms
        attribute (in that order). 

        Args
        ----
        data (EbbFlowGameData instance): Game data object to be transformed.

        Returns
        -------
        data (EbbFlowGameData instance): Game data object after the 
            sequence of transformations is applied.
        """
        for t in self.transforms:
            data = t(data)
        return data

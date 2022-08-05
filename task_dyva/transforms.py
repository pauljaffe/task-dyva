"""Data transformations defined as classes with an implemented
__call__ method. These can be supplied as inputs to the TaskDataset class.
"""
import math

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

    'gaussian': Responses are smoothed with a Gaussian kernel with 
        SD (width) set by the key 'kernel_sd' in params.
    'adaptive_gaussian': As above, but the SD (width) of the kernel depends
        on the type of trial. Each trial is assigned to one of the following
        four trial types: congruent/stay, congruent/switch, 
        incongruent/stay, and incongruent/switch. The kernel SD for a given
        trial is set to the SD of the RT distribution for the corresponding 
        trial type. 
    'kde': As above, but instead of smoothing with a Gaussian, the smoothing
        kernels are kernel density estimates of the RT distributions for
        each of the four trial types. 
    """

    kernel_t_max = 3520

    def _build_sm_kernel(self, params):
        t = np.arange(0, self.kernel_t_max, params['step_size'])
        if params['smoothing_type'] == 'gaussian':
            sd = params['kernel_sd']
            m = self.kernel_t_max / 2
            single_kernel = np.exp(-0.5 * ((t - m) / sd) ** 2)
            kernel = [single_kernel / np.amax(single_kernel) 
                      for i in range(4)]
        elif params['smoothing_type'] == 'adaptive_gaussian':
            std_devs = params['params']
            m = self.kernel_t_max / 2
            kernels_unnorm = [np.exp(-0.5 * ((t - m) / std_devs[k]) ** 2)
                              for k in range(4)]
            kernel = [k / np.amax(k) for k in kernels_unnorm]
        elif params['smoothing_type'] == 'ex_gauss':
            dists = [params['ex_gauss_rv'] for i in range(4)]
            kernel = self._norm_and_shift(dists, t)
        elif params['smoothing_type'] == 'ex_gauss_by_trial_type':
            kernel = self._norm_and_shift(params['params'], t)
        elif params['smoothing_type'] == 'kde':
            kernel = self._norm_and_shift(params['params'], t)
        elif params['smoothing_type'] == 'optimal':
            # Smooth with a rectangular kernel
            rt = params['remap_rt']
            resp_buffer = params['post_resp_buffer']
            min_rt = params['optimal_min_rt']
            half_win = (rt+resp_buffer) / 2
            t_on = self.kernel_t_max/2 + min_rt - rt
            t_off = self.kernel_t_max/2 + 2*half_win - rt
            on_ind = int(t_on // params['step_size'])
            off_ind = int(t_off // params['step_size'])
            single_kernel = np.zeros(len(t))
            single_kernel[on_ind:off_ind] = 1
            kernel = [single_kernel for i in range(4)]
        self.kernel = kernel

    def _norm_and_shift(self, dists, t):
        pdfs = [dists[k].pdf(t) for k in range(4)]
        pdfs_norm = [p / sum(p) for p in pdfs]
        means = [np.dot(pn, t) for pn in pdfs_norm]
        shifts = [self.kernel_t_max / 2 - m for m in means]
        kernel = [dists[k].pdf(t - s) / np.amax(dists[k].pdf(t - s))
                  for k, s in zip(range(4), shifts)]
        return kernel

    def __call__(self, data):
        """Smooth the responses of a single game.

        Args
        ----
        data (EbbFlowGameData instance): Game data object to be transformed.

        Returns
        -------
        data (EbbFlowGameData instance): Game data object after the 
            transformation is applied. 
        """

        responses = data.continuous['urespdir']
        sm_responses = np.zeros(responses.shape[:2])
        for direction in range(responses.shape[1]):
            for trial_type in range(responses.shape[2]):
                conv_full = np.convolve(responses[:, direction, trial_type], 
                                        self.kernel[trial_type], mode='same')
                sm_responses[:-1, direction] += conv_full[1:]
        data.continuous['urespdir'] = sm_responses
        return data


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

# Test the LLDVAE implementation
import os

import torch
import pytest

from task_dyva.testing import SetUpTests, elbo_testing


@pytest.mark.parametrize('resamp, sm, noise_sd, rt_method, out_method, \
                          out_thresh', [
    (None, 'gaussian', 0, 'max', 'mad', 10),
])
def test_model(resamp, sm, noise_sd, rt_method, out_method, out_thresh):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    rel_data_dir = 'user1365'
    raw_data_dir = os.path.join(this_dir, 'test_data', rel_data_dir)
    raw_data_fn = 'user1365.pickle'
    test_dir = raw_data_dir

    tparams = {'data_augmentation_type': resamp, 'smoothing_type': sm,
               'noise_sd': noise_sd, 'rt_method': rt_method,
               'start_times': [5000, 55000],
               'upscale_mult': 2, 'duration': 5000}

    expt_kwargs = {'outlier_method': out_method, 'outlier_thresh': out_thresh,
                   'mode': 'full', 'nth_play_range': [150, 200], 
                   'train': tparams, 'val': tparams, 'test': tparams}

    tester = SetUpTests(test_dir, raw_data_dir, raw_data_fn, **expt_kwargs)
    tester.tear_down(test_dir)
    expt = tester.make_experiment()
    split_datasets = [expt.train_dataset, expt.val_dataset, expt.test_dataset]

    for dataset in split_datasets:
        check_output_shape(expt, dataset)
        check_sample_independence(expt, dataset)
        check_dead_subgraphs(expt, dataset)


def check_dead_subgraphs(expt, dataset):
    # Make sure all parameters get updated
    expt.train()
    for param in expt.parameters():
        param.grad = None

    loss, x = elbo_testing(expt.model, dataset.xu, expt.anneal_param)
    mloss = loss.mean(0)
    mloss.backward()
    expt.optimizer.step()

    for param_name, param in expt.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            grad_sum = torch.sum(param.grad ** 2).item()
            assert grad_sum != 0.


def check_sample_independence(expt, dataset, mask_inds=[0, 13, 23]):
    # Forward pass, mask loss for samples in mask_inds
    expt.train()
    inputs = dataset.xu
    inputs.requires_grad = True
    loss, x = elbo_testing(expt.model, inputs, expt.anneal_param)
    mask = torch.ones_like(loss)
    mask[mask_inds] = 0
    loss = loss * mask

    # Backward pass
    mloss = loss.mean(0)
    mloss.backward()

    # Check gradient is zero for masked samples
    for i in range(inputs.shape[1]):
        grad = inputs.grad[:, i, :]
        if i in mask_inds:
            assert torch.all(grad == 0).item()
        else:
            assert torch.all(grad != 0).item()


def check_output_shape(expt, dataset):
    expt.eval()
    xu = dataset.xu
    n_timesteps = xu.shape[0]
    n_samples = xu.shape[1]
    w_dim = expt.config_params['model_params']['w_dim']
    z_dim = expt.config_params['model_params']['latent_dim']

    outputs = expt.model.forward(xu, generate_mode=True, clamp=False)
    x_in = outputs[0]
    assert x_in.shape == (n_timesteps, n_samples, 4)
    rates = outputs[1].mean
    assert rates.shape == (n_timesteps, n_samples, 4)
    w = outputs[2]
    assert w.shape == (n_timesteps, n_samples, w_dim)
    z = outputs[3]
    assert z.shape == (n_timesteps, n_samples, z_dim)
    w_means = outputs[4]
    assert w_means.shape == (1, n_samples, w_dim)
    w_vars = outputs[5]
    assert w_vars.shape == (1, n_samples, w_dim)

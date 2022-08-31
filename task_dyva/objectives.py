import torch


def elbo(model, xu, anneal_param):
    """Calculate the evidence lower bound (ELBO) objective.

    Args
    ----
    model (object): Model instance for which to calculate the ELBO.
        The model must subclass torch.nn.Module and have a forward
        method implemented.
    xu (PyTorch tensor): Dataset for which to calculate the ELBO.
    anneal_param (single element PyTorch tensor): The annealing parameter used
        to calculate the ELBO. This should be set to a value between
        zero and one.

    Returns
    -------
    avg_log_like (single element PyTorch tensor): The negative log-likelihood
        of the model's responses, averaged over the batch.
    loss (single element PyTorch tensor): The negative ELBO, averaged over
        the batch.
    """

    x, px_w, w, _, w_means, w_vars = model(xu)
    log_like = (px_w.log_prob(x)).sum(0).sum(-1)
    lqw_x = model.qw_x(w_means, w_vars).log_prob(w).sum(0).sum(-1)
    lpw = model.pw(*model.pw_params).log_prob(w).sum(0).sum(-1)
    loss = -(anneal_param * log_like + anneal_param * lpw - lqw_x).mean(0)
    avg_log_like = -log_like.mean(0)
    return avg_log_like, loss

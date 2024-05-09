import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.stochastic_samplers import StochasticSampler 
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import PosteriorMean
from botorch.utils.transforms import unnormalize

from .gp import SingleTaskGP, MultiTaskGP 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model_name, model_args, input_dim, output_dim):
    if model_name == 'gp':
        if output_dim == 1:
            return SingleTaskGP(model_args, input_dim, output_dim)
        else:
            return MultiTaskGP(model_args, input_dim, output_dim)
    else:
        raise NotImplementedError("Model type %s does not exist" % model_name)

def initialize_points(bounds, n_init_points, device):
    if n_init_points < 1:
        init_x = torch.zeros(1, 1).to(device)
    else:
        bounds = bounds.to(device, dtype=torch.double)
        init_x = draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)

    return init_x

def construct_acqf_by_model(model, train_x, train_y, beta=1.0, num_objectives=1):
    dim = train_y.shape[1]
    sampler = StochasticSampler(sample_shape=torch.Size([256]))
    if num_objectives==1:
        acqf = qUpperConfidenceBound(model=model, beta=beta, sampler=sampler)
    else:
        weights = torch.ones(dim)/dim
        posterior_transform = ScalarizedPosteriorTransform(weights.to(train_x))
        acqf = qUpperConfidenceBound(model=model, 
        beta=beta, 
        sampler=sampler,
        posterior_transform = posterior_transform
        )

    return acqf 

def get_current_best(model, bounds, q=1):
    # obtain the current best from the model using posterior
    normalized_opt_x, _ = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=bounds,
            q=q,
            num_restarts=64,
            raw_samples=1024, 
            sequential=False,
        )
    with torch.no_grad():
        posterior = model.posterior(torch.tensor(normalized_opt_x).to(device))
        posterior_mean = posterior.mean.cpu().numpy()
    opt_x = unnormalize(normalized_opt_x, bounds).numpy()
    print('Best estimate %s with predicted score : %s'%(opt_x, posterior_mean))

    return posterior_mean, opt_x
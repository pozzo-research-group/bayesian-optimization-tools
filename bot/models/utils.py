import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.stochastic_samplers import StochasticSampler 
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from .gp import SingleTaskGP, MultiTaskGP 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model_name, model_args, input_dim, output_dim, device):
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

def construct_acqf_by_model(model, train_x, train_y, num_objectives=1):
    dim = train_y.shape[1]
    sampler = StochasticSampler(sample_shape=torch.Size([256]))
    if num_objectives==1:
        acqf = qUpperConfidenceBound(model=model, beta=100, sampler=sampler)
    else:
        weights = torch.ones(dim)/dim
        posterior_transform = ScalarizedPosteriorTransform(weights.to(train_x))
        acqf = qUpperConfidenceBound(model=model, 
        beta=100, 
        sampler=sampler,
        posterior_transform = posterior_transform
        )


    return acqf 
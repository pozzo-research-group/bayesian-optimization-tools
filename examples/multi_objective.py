import os, shutil, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

from bot.models.utils import initialize_model, initialize_points
from bot.datatools.base import FunctionValuedExperiment 

parser = argparse.ArgumentParser(
                    prog='Example of single objective campaign using BOT',
                    description='Performs optimization of a Gaussian curve',
                    epilog='...')
parser.add_argument('itr', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.itr # specify the current itereation number
print("Running the campagin for itereation %d"%ITERATION)
# hyper-parameters
BATCH_SIZE = 4
N_INIT_POINTS = 5
MODEL_NAME = "gp"
DESIGN_SPACE_DIM = 2
OUTPUT_DIM = 2

EXPT_DATA_DIR = "./MO/data/"
SAVE_DIR = './MO/output/'
PLOT_DIR = './MO/plots/'

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.makedirs(direc)


""" Set up design space bounds """
design_space_bounds = [(-10, 10), (0.1,1.0)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

""" Create a GP model class for surrogate """
model_args = {"model":"gp",
"num_epochs" : 1000,
"learning_rate" : 1e-3
}

""" Helper functions """
def featurize(expt):
    """ Featurize spectras data.

    This functions takes in the spectra and featurize them into
    mean and standard deviation.

    We use these features to run a mulit-objective optimization.

    """
    num_samples = expt.spectra.shape[0]
    target_spectra = generate_spectra([[-2,0.5]])[1].squeeze()
    target_mean, target_std = target_spectra.mean(), target_spectra.std()
    print(target_spectra.shape)
    train_x = torch.from_numpy(expt.comps).to(device)
    train_y = []
    for i in range(num_samples):
        query_mean, query_std = expt.spectra[i,:].mean(), expt.spectra[i,:].std()
        train_y.append([query_mean-target_mean, query_std-target_std])

    train_y = np.asarray(train_y)
    train_y = torch.from_numpy(train_y).to(device)

    return train_x, train_y.reshape(num_samples, OUTPUT_DIM)

def run_iteration(expt):
    r"""Perform a single iteration of active phasemapping.

    helper function to run a single iteration given 
    all the compositions and spectra obtained so far. 
    """
    # assemble data for surrogate model training  
    comps_all = expt.comps 
    spectra_all = expt.spectra 
    print('Data shapes : ', comps_all.shape, spectra_all.shape)

    standard_bounds = torch.tensor([(float(1e-5), 1.0) for _ in range(DESIGN_SPACE_DIM)]).transpose(-1, -2).to(device)
    gp_model = initialize_model(MODEL_NAME, model_args, DESIGN_SPACE_DIM, OUTPUT_DIM, device) 

    train_x, train_y = featurize(expt)
    print(train_x.shape, train_y.shape)
    normalized_x = normalize(train_x, bounds)
    gp_model = gp_model.fit(normalized_x, train_y)
    torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model_%d.pt'%ITERATION)

    # use multi-objective acqusition
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acquisition = qNoisyExpectedHypervolumeImprovement(
        model=gp_model,
        ref_point=[0.0,0.0], 
        X_baseline=normalize(train_x, bounds),
        prune_baseline=True, 
        sampler=sampler,
    )

    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, 
        standard_bounds, 
        q=BATCH_SIZE, 
        num_restarts=20, 
        raw_samples=1024, 
        return_best_only=True,
        sequential=False,
        options={"batch_limit": 1, "maxiter": 10, "with_grad":True}
        )

    # calculate acquisition values after rounding
    new_x = unnormalize(normalized_candidates.detach(), bounds=bounds) 

    torch.save(train_x.cpu(), SAVE_DIR+"train_x_%d.pt" %ITERATION)
    torch.save(train_y.cpu(), SAVE_DIR+"train_y_%d.pt" %ITERATION)

    return new_x.cpu().numpy(), gp_model, acquisition, train_x

def generate_spectra(comps):
    "This functions generates Gaussian functions given mean and std"

    print("Generating spectra for iteration %d"%ITERATION, '\n', comps)
    n_domain = 100
    spectra = np.zeros((len(comps), n_domain))
    lambda_ = np.linspace(-5,5,num=n_domain)
    for j, cj in enumerate(comps):
        scale = 1/(np.sqrt(2*np.pi)*cj[1])
        spectra[j,:] =scale*np.exp(-np.power(lambda_ - cj[0], 2.) / (2 * np.power(cj[1], 2.)))

    return lambda_, spectra

# Set up a synthetic data emulating an experiment
if ITERATION == 0:
    init_x = initialize_points(bounds, N_INIT_POINTS, device)
    comps_init = init_x.detach().cpu().numpy()
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
    wl, spectra = generate_spectra(comps_init)
    np.save(EXPT_DATA_DIR+'stimuli.npy', wl)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)
else: 
    expt = FunctionValuedExperiment(EXPT_DATA_DIR, ITERATION)
    fig, ax = plt.subplots()
    expt.plot(ax, design_space_bounds)
    plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
    plt.close()

    # obtain new set of compositions to synthesize and their spectra
    comps_new, gp_model, acquisition, train_x = run_iteration(expt)
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), comps_new)
    _, spectra = generate_spectra(comps_new)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)
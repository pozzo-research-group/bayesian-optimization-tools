from abc import ABC, abstractmethod
import numpy as np 
from scipy import interpolate 
import pandas as pd 

FUNCTION_VALUED_EXPERIMENTS = ["uvvis", "saxs", "dls"]
SCALAR_VALUED_EXPERIMENTS = ["zeta"]

class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min 
        self.max = max 
        self.range = max-min

    def transform(self, x):
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        return (self.range*xt)+self.min

def scaled_tickformat(scaler, x, pos):
    return '%.1f'%scaler.inverse(x)

class ScalarValuedExperiment(ABC):
    def __init__(self, directory):
        self.dir = directory

class FunctionValuedExperiment(ABC):
    r"""Base class for function valued experiment

    This class represents experiements where the output is a function
    such as UV-Vis spectroscopy, SAXS, DLS etc.

    Inputs:
    -------
        directory: directory location of the spectral data.
            This should contains numpy files named `comps_x.npy` (shape num_samples x dim)
            where x corresponds to campaign iteration and 
            `spectra_x.npy` (shape num_samples x dim) for spectra, and `stimuli.npy` file 
            that represents the common x-axis sampling for the campaign.

            All the numpy files should represent samples in rows

        iterations: An integer representing the current campaign iteration.

    """
    def __init__(self, directory, iterations):
        self.dir = directory
        comps, spectra = [], []
        for k in range(iterations):
            comps.append(np.load(self.dir+'comps_%d.npy'%k))
            spectra.append(np.load(self.dir+'spectra_%d.npy'%k))
            print('Loading data from iteration %d with shapes:'%k, comps[k].shape, spectra[k].shape)
        self.comps = np.vstack(comps)
        self.points = self.comps
        self.spectra = np.vstack(spectra)
        self.stimuli = np.load(self.dir+'stimuli.npy')
        self.t = (self.stimuli - min(self.stimuli))/(max(self.stimuli) - min(self.stimuli))

    def normalize(self, f):
        r"""Normalize the spectra on function space using L2 norm

        inputs:
        -------
            f : function data to be normalized (shape (n_domain, ), )
        
        """
        norm = np.sqrt(np.trapz(f**2, self.stimuli))

        return f/norm 

    def plot(self, ax, bounds):
        r"""Pretty plot of 2D design spaces with spectra embedded

        inputs:
        -------
            ax : matplotlib.pyplot axis object
            bounds : design space bounds as a list consisting of [min, max]*dim
        
        """
        if not self.comps.shape[1]==2:
            raise RuntimeError("Plotting only works for 2-dimensional spaces")
        bounds = np.asarray(bounds).T
        scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
        scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
        ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
        ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
        for i, (ci, si) in enumerate(zip(self.comps, self.spectra)):
            norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
            self._inset_spectra(norm_ci, self.t, si, ax)

        return 

    def _inset_spectra(self, c, t, ft, ax, **kwargs):
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(t, ft, **kwargs)
        ins_ax.axis('off')
        
        return 

    def spline_interpolate(self, y):
        r"""Spline interpolator for functions

        inputs:
        -------
            y : function data to be normalized (shape (n_domain, ), )
        
        """
        spline = interpolate.splrep(self.stimuli, y, s=0)
        wl_ = np.linspace(min(self.stimuli), max(self.stimuli), num=100)
        I_grid = interpolate.splev(wl_, spline, der=0)
        norm = np.sqrt(np.trapz(I_grid**2, wl_)) 

        return I_grid/norm 
    
    def spectra_post_process(self, type):
        r"""Post-processing function for spectra

        inputs:
        -------
            type : string representing two types of post-processing available
                ["interpolate" or "normalize"]
        
        """
        if type=="interpolate":
            for i, si in enumerate(self.spectra):
                self.spectra[i,:] = self.spline_interpolate(si)
        elif type=="normalize":
            for i, si in enumerate(self.spectra):
                self.spectra[i,:] = self.normalize(si)            



        
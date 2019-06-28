
####################################################
########## Magnetometry related definitions ########
####################################################

import sys, os, copy, warnings, time
import qinfer as qi
import numpy as np
from copy import deepcopy
from functools import partial

gamma = 28.0247
PI = np.pi

def radfreq_to_B(radfreq):
    # radfreq: in rad*MHz
    # B: in uT
    radfreq = np.array([radfreq])
    if len(radfreq) is 1:
        return (radfreq / (2*PI) / gamma * 1000)[0]
    else:
        B = list(lambda fq: fq / (2*PI) / gamma * 1000, fq in radfreq)
        return B

def B_to_radfreq(B):
    # radfreq: in rad*MHz
    # B: in uT
    B = np.array([B])
    if len(B) is 1:
        return (B * 2*PI * gamma / 1000)[0]
    else:
        radfreq = list(lambda eachB: eachB * 2*PI * gamma / 1000, eachB in B)
        return radfreq
        
####################################################
########## I/O functions ###########################
####################################################     

import pickle
import datetime   
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib import cm as colmap
from matplotlib.patches import Ellipse

def rescaledata(rawdata, offset = 0.):
    # rawdata is numpy array
    # offset sets the required offset for probabilities (if any)
    if len(np.shape(rawdata)) > 1 or str(type(rawdata)) != "<class 'numpy.ndarray'>":
        raise Exception("rawdata must be 1D numpy array")
    
    rescale_factor = max(rawdata)-min(rawdata)
    
    return (rawdata - min(rawdata) + offset)/rescale_factor
    
    
def rescalefewdata(rawdata, offset = 0.5):
    # rawdata is numpy array
    # offset sets the required offset for probabilities (if any)
    if len(np.shape(rawdata)) > 1 or str(type(rawdata)) != "<class 'numpy.ndarray'>":
        raise Exception("rawdata must be 1D numpy array")
    
    datamean = np.mean(rawdata)
    
    rescale_upfactor = max(rawdata)-datamean
    rescale_downfactor = datamean-min(rawdata)
    
    rescaledata = []
    
    for datum in rawdata:
        if datum > datamean:
            rescaledata.append( offset + (datum-datamean)/2/rescale_upfactor )
        else:
            rescaledata.append( offset - (datamean-datum)/2/rescale_downfactor )
    
    return np.array(rescaledata)
    

def rescaletwodata(rawdata, offset = 0.5):
    # rawdata is numpy array
    # offset sets the required offset for probabilities (if any)
    if len(np.shape(rawdata)) > 1 or str(type(rawdata)) != "<class 'numpy.ndarray'>":
        raise Exception("rawdata must be 1D numpy array")
    
    datamean = np.mean(rawdata)
    
    rescale_upfactor = max(rawdata)-datamean
    rescale_downfactor = datamean-min(rawdata)
    
    rescaledata = []
    
    for datum in rawdata:
        if datum >= datamean:
            rescaledata.append( offset + (datum-datamean)/2/rescale_upfactor )
        else:
            rescaledata.append( offset - (datamean-datum)/2/rescale_downfactor )
    
    return np.array(rescaledata)
    
    
def adjust_data_qi(xydata, n_shots = 1, omegabounds = [0, 15]):
    """
    n_shots: number of repetitions for each experiment in the data
    [omega_min, omega_max]: bounds for the frequency parameter
    """
    
    [Ramsey_xdata, Ramsey_ydata] = xydata
    [omega_min, omega_max] = omegabounds
    
    round_ydata = np.round(Ramsey_ydata, int(np.log10(n_shots))) #we need integer "simulated" counts
    adjdata = np.column_stack([n_shots * round_ydata, Ramsey_xdata, n_shots * np.ones_like(Ramsey_ydata)])
    
    return adjdata
    
def mytimestamp():
    timestamp = str(datetime.datetime.now()).split('.')[0]
    timestamp = "_"+timestamp.replace(" ", "_")
    timestamp = timestamp.replace(":", ".")
    return(timestamp)

def savedata(dire, thisfile, data, fileformat = ".pk"):
    if not os.path.exists(dire):
        os.makedirs(dire)
    outpath = os.path.normpath(dire+"/"+thisfile+fileformat)
    
    if fileformat == ".pickle" or fileformat == ".pk":
        with open(outpath, "wb") as f:
            pickle.dump(data, f)
            
    elif fileformat == '.mat':
        savemat(outpath, mdict={'out': npoutput}, oned_as='row')
        
    else: 
        with open(dire+"\\"+thisfile+fileformat, 'wb') as outfile:
            for slice_2d in data[0]:
                np.savetxt(outfile, slice_2d)
    
    print(outpath)

def saveresults(dire, thisfile, string_output):
    textfile = open(dire+"\\"+str(thisfile), "w")

    for item in string_output:
        textfile.write(item)
        textfile.write("\n")
    
    textfile.close()

    
    
def readdata(thisfile):
    outpath = os.path.normpath(thisfile)
    with open(outpath, "rb") as f:
        data = pickle.load(f)
    return data
    
def savefigs(dire, thisfile):
    """
    dire: is the directory where the file will be stored, non terminated by slashes
        e.g. "C:\\mydirectory"
    thisfile: is the name of the file with extension, without slashes
        e.g. "myfile.pdf"
    """
    if not os.path.exists(dire):
        os.makedirs(dire)
    outpath = os.path.normpath(dire+"/"+thisfile)
    plt.savefig(outpath, bbox_inches='tight')
    print(outpath)
    
    
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
    
####################################################
########## Standard Data Analysis ##################
#################################################### 

from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.stats import norm

def Lorentzian(x, x0, gamma, norm_factor):
    return gamma / ( 2*PI* ( (x-x0)**2 + (gamma/2)**2 )  ) / norm_factor 

def Gaussian(x, mean = 0., sigma = 1.):
    return norm.pdf(x, loc = mean, scale = sigma)
    
    
def dist_normalize(spectrum, method = "max", bounds = (-np.inf, np.inf), args = None):
    """
    spectrum: must be an array for the "max" method, or a Python callable for "area"
    """
    
    if method is "max":
        spectrum /= max(spectrum)
        
    elif method is "area":
        area = integrate.quad(spectrum, bounds[0], bounds[1], args= args[1:]  )[0]
        spectrum = spectrum(*args)
        spectrum /= area
  
    return spectrum
    

def fft_prec(data, sampling_factor = 1):
    [Ramsey_xdata, Ramsey_ydata] = data
    
    sampling = int(sampling_factor*len(Ramsey_ydata))
    
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(Ramsey_ydata - Ramsey_ydata.mean(), n=sampling)))**2
    ft_freq = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n=sampling, d=Ramsey_xdata[1] - Ramsey_xdata[0]))
    
    est_omega = np.abs(ft_freq[spectrum.argmax()])
    
    xlimit = 2*est_omega
    spectrum = np.abs(spectrum/np.sum(spectrum))
    peak_gamma = 2/max(spectrum)/PI
    
    fft_gamma = (ft_freq[1]-ft_freq[0])*sampling_factor
    
    norm_factor = peak_gamma / (ft_freq[1]-ft_freq[0]) / sampling_factor
    p0 = [est_omega, fft_gamma, norm_factor]
    try:
        Lore_fit = curve_fit(Lorentzian, ft_freq, spectrum, p0=p0)
        return [ft_freq, spectrum, est_omega, xlimit, Lore_fit[0][1], Lore_fit[0][2] ]
    except:
        print("Lorentzian fit failed")
        return [ft_freq, spectrum, est_omega, xlimit, fft_gamma, norm_factor ]
        
        
        
def fft_fitonly(data, sampling_factor = 1):
    [Ramsey_xdata, Ramsey_ydata] = data
    
    sampling = int(sampling_factor*len(Ramsey_ydata))
    
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(Ramsey_ydata - Ramsey_ydata.mean(), n=sampling)))**2
    ft_freq = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n=sampling, d=Ramsey_xdata[1] - Ramsey_xdata[0]))
    
    est_omega = np.abs(ft_freq[spectrum.argmax()])
    
    xlimit = 2*est_omega
    spectrum = np.abs(spectrum/np.sum(spectrum))
    peak_gamma = 2/max(spectrum)/PI
    
    fft_gamma = (ft_freq[1]-ft_freq[0])*sampling_factor
    
    norm_factor = peak_gamma / (ft_freq[1]-ft_freq[0]) / sampling_factor
    p0 = [est_omega, fft_gamma, norm_factor]
    try:
        Lore_fit = curve_fit(Lorentzian, ft_freq, spectrum, p0=p0)
        return Lore_fit
    except:
        print("Lorentzian fit failed")
        return -1
    

def retrieve_experiment(data, experiment):
    
    heu_time = experiment[0][0] 
    idx = (np.abs(data[:, 1]-heu_time)).argmin()
    
    if data[idx][2] > 1:
        prob = data[idx][0]/data[idx][2]
        datum = np.random.choice(np.array([0,1]), p = (1-prob, prob))
    else:
        datum = data[idx][0]
    
    return [ datum, data[idx][1] ] #respectively the datum and time found 
    

def Poissonian_step(list_length, previous_idx = None, seed = 0.95, probs_array = None):
    if (len(probs_array) is not list_length) and (probs_array is not None):
        raise Exception("DimensionError: list_length must equate probs_list")
    
    if previous_idx is None:
        previous_idx = probs_array.index(max(probs_array))
    chosen_idx = np.random.choice(np.arange(list_length), p=probs_array)
    new_probs_array = copy.deepcopy(probs_array)
    
    reshuffle = (1-seed)/len(new_probs_array)
    
    if previous_idx == chosen_idx:
        new_probs_array[chosen_idx] *= seed
        new_probs_array += reshuffle
#         print(new_probs_array)
        new_probs_array[chosen_idx] -= reshuffle
    else:
        new_probs_array = np.zeros(list_length)
        new_probs_array[chosen_idx] += seed      
        
    #print(new_probs_array)

    new_probs_array = new_probs_array/np.sum(new_probs_array)

    return (chosen_idx, new_probs_array)    



####################################################
########## Additional Data Analysis ##################
####################################################    

   
    
def DampedOscill(t, Omega, invT2):
    y = 1- np.exp(-t*invT2) * np.cos(Omega * t / 2) ** 2 - 0.5*(1-np.exp(-t*invT2))
    return(y)

    
    
def sweepscale_function(sweeps, *kwargs):
    if len(kwargs) is 3:
        c = kwargs[2]
    else:
        c = 0.
    
    return kwargs[0]*( (1.0*sweeps)**kwargs[1] ) + c
    
  
  
def scale_function(acctime, a, b):
    return a*(acctime**b)
    
    
def scaling_fit(scale_fit, acctimes, medcov, errcov, p0 = None, skip_start = 2, skip_end = -1):
    if errcov is not None:
        sigma = errcov[skip_start:skip_end]
    else:
        sigma = None
    popt, pcov = curve_fit(scale_function, acctimes[skip_start:skip_end], medcov[skip_start:skip_end], p0 = p0, sigma = sigma)
    if popt[1] < -1.0: popt[1] = -1.0
    return popt, pcov
    
    
def time_analysis(npoutput, out_index, convert_factor = 1., plot_color=None, plot_title="", 
                    enable_save=False, todir=None, tofile = None):
    all_pgh_times =  []
    
    for particle in range(len(npoutput[0,:])):
        all_pgh_times.extend(npoutput[0,particle,out_index,:].tolist())
        
    medtimes = np.mean(npoutput[0,:,out_index,-1])*convert_factor
    print("Last (average) selected time ms: " + str(medtimes) )
    
    #plt.figure(figsize=(300, 300))
    
#     plt.grid(True)
    
    MIN, MAX = min(all_pgh_times), max(all_pgh_times)
    
    plt.hist(np.array(all_pgh_times), log=True, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50) ,
        color = plot_color, edgecolor='black')

    plt.xlabel(r'Times $\tau$ ($\mu$s)')
    plt.ylabel('Occurrences')
    
    plt.title(plot_title)
    plt.gca().set_xscale("log")
    
    if enable_save: 
        if todir is None or tofile is None:
            Exception("A file and path are needed to enable saving")
        else:
            savefigs(todir, tofile)
    
    plt.show()

    return [ np.array(all_pgh_times)*convert_factor, medtimes ]



 
    
  
  
####################################################
########## QHL definitions       ##################
#################################################### 


    
class ExpPrecessionModel(qi.SimplePrecessionModel):
    r"""
    """
        
    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('w_', 'float')]
        
        

class ExpDecoPrecessionModel(qi.FiniteOutcomeModel, qi.DifferentiableModel):
    r"""
    Model that attemprs learning unknown decoherence as a model parameter
    """
    
    ## INITIALIZER ##

    def __init__(self, min_freq=0, rescaleT2 = 1.0):
        super(ExpDecoPrecessionModel, self).__init__()
        self._min_freq = min_freq
        self._rescaleT2 = rescaleT2

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 2
    
    #T is the T2
    @property
    def modelparam_names(self):
        return ['w_', 'T_']
    
    @property
    def modelparam_dtype(self):
        return [('w_', 'float'), ('T_', 'float')]
        
    @property
    def expparams_dtype(self):
        return [('ts', 'float'), ('w_', 'float'), ('T_', 'float')]
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##
    

    
    def are_models_valid(self, modelparams):
        w_, T_ = modelparams.T
        return np.all(
            [w_ > 0, T_ > 0], axis=0)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # Customized here.
        super(ExpDecoPrecessionModel, self).likelihood(
            outcomes, modelparams, expparams
        )
        
        #print('outcomes = ' + repr(outcomes))

        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]

        #print('expparams = ' + repr(expparams))
        #print('modelparams = ' + str(modelparams[0:1]))
            
        t = expparams['ts']
        T_ = modelparams[:,1]
        dw = modelparams[:,0]
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        pr0[:, :] = (np.array([np.exp(-t*T_/10) * (np.cos(t * dw / 2) ** 2) + 
                    0.5*(1-np.exp(-t*T_/10))])).T
        
        #print("Pr0 = " + str(pr0) )

        #print("likelihoods: " + str(qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)))
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
        
        
    def score(self, outcomes, modelparams, expparams, return_L=False):
        # Pass the expparams to the superclass as a record array.
        new_eps = np.empty(expparams.shape, dtype=super(SimplePrecessionModel, self).expparams_dtype)
        new_eps['w_'] = 0
        new_eps['ts'] = expparams

        return super(SimplePrecessionModel, self).score(outcomes, modelparams, new_eps, return_L)
        
        
        
        
        
        
class ExpDecoKnownPrecessionModel(qi.FiniteOutcomeModel, qi.DifferentiableModel):
    r"""
    Model that imposes known decoherence as a user-defined parameter
    """
    
    ## INITIALIZER ##

    def __init__(self, min_freq=0, invT2 = 0.):
        super(ExpDecoKnownPrecessionModel, self).__init__()
        self._min_freq = min_freq
        self._invT2 = invT2

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    # T is the T2
    @property
    def modelparam_names(self):
        return ['w_']
    
    @property
    def modelparam_dtype(self):
        return [('w_', 'float')]
        
    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('w_', 'float')]
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    # METHODS ##
    

    
    def are_models_valid(self, modelparams):
        return np.all(modelparams > self._min_freq, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # Customized here.
        super(ExpDecoKnownPrecessionModel, self).likelihood(
            outcomes, modelparams, expparams
        )
        
        #print('outcomes = ' + repr(outcomes))

        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]

        #print('expparams = ' + repr(expparams))
        #print('modelparams = ' + str(modelparams[0:1]))
            
        t = expparams['t']
        dw = modelparams[:,0]
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        pr0[:, :] = (np.array([np.exp(-t*self._invT2) * (np.cos(t * dw / 2) ** 2) + 0.5*(1-np.exp(-t*self._invT2))])).T
        
        #print("Pr0 = " + str(pr0) )

        #print("likelihoods: " + str(qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)))
        
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
        
        
    def score(self, outcomes, modelparams, expparams, return_L=False):
        # Pass the expparams to the superclass as a record array.
        new_eps = np.empty(expparams.shape, dtype=super(SimplePrecessionModel, self).expparams_dtype)
        new_eps['w_'] = 0
        new_eps['t'] = expparams

        return super(SimplePrecessionModel, self).score(outcomes, modelparams, new_eps, return_L)
        
        

class AsymmetricLossModel(qi.DerivedModel, qi.FiniteOutcomeModel):
    """
    Model representing the case in which a two-outcome model is subject
    to asymmetric loss, such that
    
        Pr(1 | modelparams; expparams) = η Pr(1 | modelparams; expparams, no loss),
        Pr(0 | modelparams; expparams)
            = Pr(0 | modelparams; expparams, no loss) + (1 - η) Pr(1 | modelparams; expparams, no loss)
            = 1 - Pr(1 | modelparams; expparams, no loss) + (1 - η) Pr(1 | modelparams; expparams, no loss)
            = 1 - η Pr(1 | modelparams; expparams, no loss)
            = 1 - Pr(1 | modelparams; expparams).
        
    This model considers η to be *known* and given at initialization time, rather than as a model parameter to be
    estimated.
    """
    
    def __init__(self, underlying_model, eta=1.0):
        super(AsymmetricLossModel, self).__init__(underlying_model)
        self._eta = float(eta)
        
        if not (underlying_model.is_n_outcomes_constant and underlying_model.n_outcomes(None) == 2):
            raise ValueError("Decorated model must be a two-outcome model.")
    
    ## METHODS ##
    

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(AsymmetricLossModel, self).likelihood(outcomes, modelparams, expparams)
        
        pr1 = self._eta * self.underlying_model.likelihood(
            np.array([1], dtype='uint'),
            modelparams,
            expparams
        )[0, :, :]
        
        # Now we concatenate over outcomes.
        L = qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, 1 - pr1)
        assert not np.any(np.isnan(L))
        return L
        
        
        

class SinusoidalPrecessionModel(qi.SimplePrecessionModel):
    
    def __init__(self, min_freq=0, start=None, freq_rate = None, amplitude = None):
        super(SinusoidalPrecessionModel, self).__init__()
        
        self._min_freq = min_freq

        self.start = start
        self.freq_rate = freq_rate
        self.amplitude = amplitude
    
    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('w_', 'float')]
        
    def are_models_valid(self, modelparams):
        return np.all(modelparams > self._min_freq, axis=1)

    def update_timestep(self, modelparams, expparams):
        # Update magnetic field from sinusoidal rule
        if modelparams.shape[0] == 1:
            steps = np.ones(  (modelparams.shape[0], 1, expparams.shape[0]) ) * self.amplitude * np.cos(self.freq_rate*expparams[0][0])
            return self.start[:, :, np.newaxis] + steps
        # and now add a reasonable perturbation of the particles
        else:
            steps = self.amplitude * self.freq_rate * 2 * np.random.randn(modelparams.shape[0], 1, expparams.shape[0])
            return modelparams[:, :, np.newaxis] + steps



class InfidelSinusoidalPrecessionModel(SinusoidalPrecessionModel):
    
    def __init__(self, min_freq=0, start=None, freq_rate = None, amplitude = None, fid_0 = 1, fid_1 = 1):
        super(SinusoidalPrecessionModel, self).__init__()
        
        self._min_freq = min_freq
        
        self.start = start
        self.freq_rate = freq_rate
        self.amplitude = amplitude
            
        self.fid_0 = fid_0
        self.fid_1 = fid_1
            
            
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(InfidelSinusoidalPrecessionModel, self).likelihood(
            outcomes, modelparams, expparams
        )

        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
            
        t = expparams['t']
        dw = modelparams
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        pr_base = np.cos(t * dw / 2) ** 2
        
        if self.fid_0 > self.fid_1:
            pr0[:, :] = (self.fid_0-self.fid_1)/2 + (self.fid_0+self.fid_1)/2*pr_base
        else:
            pr0[:, :] = 1 - (
                            (self.fid_1-self.fid_0)/2 + (self.fid_0+self.fid_1)/2*(1-pr_base)  )
        
        # Now we concatenate over outcomes.
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
            
    
class stdPGH(qi.Heuristic):
    """

    """
    
    def __init__(self, updater, inv_field='x_', t_field='t',
                 inv_func=qi.expdesign.identity,
                 t_func=qi.expdesign.identity,
                 maxiters=10,
                 other_fields=None
                 ):
        super(stdPGH, self).__init__(updater)
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._maxiters = maxiters
        self._other_fields = other_fields if other_fields is not None else {}
        
    def __call__(self):
        idx_iter = 0
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError("PGH did not find distinct particles in {} iterations.".format(self._maxiters))
            
        eps = np.empty((1,), dtype=self._updater.model.expparams_dtype)
        eps[self._x_] = self._inv_func(x)
        eps[self._t]  = self._t_func(1 / self._updater.model.distance(x, xp)) #self._t_func(1 / self._updater.model.distance(x, xp))
        
        for field, value in self._other_fields.items():
            eps[field] = value
        
        return eps
        
        
class rootPGH(qi.Heuristic):
    """

    """
    
    def __init__(self, updater, inv_field='x_', t_field='t',
                 inv_func=qi.expdesign.identity,
                 t_func=qi.expdesign.identity,
                 root = 1,
                 n_shots = 1,
                 maxiters=10,
                 other_fields=None
                 ):
        super(rootPGH, self).__init__(updater)
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._root = root
        self._maxiters = maxiters
        self._other_fields = other_fields if other_fields is not None else {}
        
        self.eps = np.empty((1,), dtype=updater.model.expparams_dtype)
        self.eps['n_meas'] = n_shots
        
    def __call__(self):
        idx_iter = 0
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError("PGH did not find distinct particles in {} iterations.".format(self._maxiters))

        self.eps[self._x_] = self._inv_func(x)
        self.eps[self._t]  = self._t_func(1 / self._updater.model.distance(x, xp)) #self._t_func(1 / self._updater.model.distance(x, xp))
        self.eps[self._t] = np.power(self.eps[self._t], self._root)
        
        for field, value in self._other_fields.items():
            self.eps[field] = value
        
        return self.eps
            
        
class multiPGH(qi.Heuristic):
    
    def __init__(self, updater, oplist=None, norm='Frobenius', inv_field='x_', t_field='t',
                 inv_func=qi.expdesign.identity,
                 t_func=qi.expdesign.identity,
                 maxiters=10,
                 other_fields=None
                 ):
        super(multiPGH, self).__init__(updater)
        self._oplist = oplist
        self._norm = norm
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._maxiters = maxiters
        self._other_fields = other_fields if other_fields is not None else {}
        
    def __call__(self):
        idx_iter = 0
        
        slow_PGH = 0.01   #should normally be ~1
        
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError("PGH did not find distinct particles in {} iterations.".format(self._maxiters))
            
        #print('Selected particles: #1 ' + repr(x) + ' #2 ' + repr(xp))
            
        eps = np.empty((1,), dtype=self._updater.model.expparams_dtype)
        
        idx_iter = 0 # modified in order to cycle through particle parameters with different names
        for field_i in self._x_:
            eps[field_i] = self._inv_func(x)[0][idx_iter]
            idx_iter += 1
        if self._oplist is None:   #Standard QInfer geom distance
            eps[self._t]  = self._t_func(1 / self._updater.model.distance(x, xp))
        else:
            deltaH = getH(x, self._oplist)-getH(xp, self._oplist)
            if self._norm=='Frobenius':
                eps[self._t] = slow_PGH/np.linalg.norm(deltaH)   #Frobenius norm
            elif self._norm=='MinSingVal':
                eps[self._t] = slow_PGH/minsingvalnorm(deltaH)   #Min SingVal norm
            elif self._norm=='SingVal':
                eps[self._t] = slow_PGH/singvalnorm(deltaH)   #Max SingVal
            else:
                eps[self._t] = slow_PGH/np.linalg.norm(deltaH)
                raise RuntimeError("Unknown Norm: using Frobenius norm instead")
        for field, value in self._other_fields.items():
            eps[field] = value
        
        return eps
        

class stepwise_heuristic():
    
    def __init__( self, updater, n_shots=1, max_n_experiments=500, ts = None ):
        
        if ts is None:
            ts = np.arange(1, 1 + max_n_experiments) / (2 * omega_max)
        self.expparams = np.empty((1,), dtype=updater.model.expparams_dtype)
        self.expparams['n_meas'] = n_shots
        
        self.t_iter = iter(ts)
    
    def __call__( self ):
        self.expparams['t'] = next(self.t_iter)
        return self.expparams          
        
        
        
        
def eval_loss(
        model, est_mean, true_mps=None,
        true_model=None, true_prior=None
    ):
    
    if true_model is None:
        true_model = model

    if true_mps is None:
        true_mps = true_model.update_timestep(
            promote_dims_left(true_mps, 2), expparams
        )[:, :, 0]

    if model.n_modelparams != true_model.n_modelparams:
        raise RuntimeError("The number of Parameters in True and Simulated model are different.")
                           
    n_pars = model.n_modelparams

    delta = np.subtract(*qi.perf_testing.shorten_right(est_mean, true_mps))
    loss = np.dot(delta**2, model.Q[-n_pars:])

    return loss
    



    
class TrackPrecModel():
    r"""
    
    """
    
    ## INITIALIZER ##

    def __init__(self, freq_min=0.0, freq_max=1.0, n_particles=1000, timedep = "Sin", start = None, freq_rate = None, amplitude = None, noise = None, fids = (1,1)):
        
        if timedep is "Sin":
            self.model = SinusoidalPrecessionModel(min_freq = freq_min, start=start, freq_rate = freq_rate, amplitude = amplitude)    
        elif timedep is "InfidSin":
            self.model = InfidelSinusoidalPrecessionModel(min_freq = freq_min, start=start, freq_rate = freq_rate, amplitude = amplitude, fid_0=fids[0], fid_1=fids[1]) 
        else:
            raise("Probably yet to implement...")
            
            
        self.n_particles = n_particles
        
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        self.start = start
        self.noise = noise
    
    

    def est_prec(self, data, n_shots = 1, n_experiments = 50, resample_a=None, resample_thresh=0.5, verbose=False, TBC = False):
        """
        Class for (decohered) Ramsey fringe learning
        
        data: set of [time, likelihood] data, or "None" if you want simulation
        n_experiments: number of experiments to be performed before interruption
        resample_a: provides an indicator of how much the Liu-West resampler is allowed to "move" the distribution in case of resampling
            >> higher "a" indicatively leads to slower learning, but smoother learning
        resample_thresh: provides an indicator of how frequently to call resampling. 
            >> higher the "thresh" value, more frequently resampling events will occur, indicatively leading to faster but less smooth learning
        """
        
        if TBC:
            if verbose: print("Continuing from...", self.prior)
            if verbose: print("Exparam = ", self.expparams[0][0] )
            

        else:
            self.prior = qi.UniformDistribution([self.freq_min, self.freq_max])
            self.updater = qi.SMCUpdater(self.model, self.n_particles, self.prior, resample_a=resample_a, resample_thresh=resample_thresh)
            self.heuristic = stdPGH(self.updater, inv_field='w_')          
            
            self.expparams = np.empty((1, ), dtype=self.model.expparams_dtype) 
                    
            self.lastexp = 0
            
        
        if data is None:
            self.sim = True
            if verbose: print("Simulated run")
            
            if TBC:
                if verbose: print("Last true_param: ", self.true_params)
            else:
                self.true_params = self.start
            
        else:
            warnings.warn("Placeholder function, to be debugged")
            self.sim = False
            if verbose: print("Experimental run")


            if ( self.true_params[0][0] >= self.prior._ranges[0][1] or self.true_params[0][0] <= self.prior._ranges[0][0] ):
                warnings.warn("Chosen prior appears incompatible with FFT analysis, consider modifying the range of the prior distribution")
            
            mydata = adjust_data_qi(data, n_shots = n_shots, omegabounds = [self.freq_min, self.freq_max])
        
        true_trajectory = []
        est_trajectory = []
        
        track_loss = np.empty(n_experiments)
        track_cov = np.empty(n_experiments)
        track_time = np.empty(n_experiments)
        track_pgh_time = np.empty(n_experiments)
        track_acctime = np.zeros(n_experiments)

        for idx_experiment in range(n_experiments):
            experiment = self.heuristic()     

            self.expparams[0][0] = idx_experiment + self.lastexp
            
            track_pgh_time[idx_experiment] = deepcopy(experiment[0][0])
            
            if verbose: print("Proposed experiment ", experiment)
            
            if self.sim:
                if self.noise is None:
                    datum = self.model.simulate_experiment(self.true_params, experiment)
                elif self.noise is 'Binomial':
                    pr0 = self.model.likelihood(0, self.true_params, experiment)
                    datum = np.random.binomial(1, p=1-pr0)
            else:
                [datum, newtime] = retrieve_experiment(mydata, experiment)
                track_time[idx_experiment] = newtime
                experiment[0][0] = newtime
                if verbose: print("Found experiment ", experiment)
            
            track_acctime[idx_experiment] = track_acctime[max(0, idx_experiment-1)] + experiment[0][0]
            
            if verbose: print("Datum", datum)
            
            self.updater.update(datum, experiment)
            
            self.true_params = self.model.update_timestep(self.true_params, self.expparams)[:, :, 0]
            # print(self.model.update_timestep(self.true_params, self.expparams))
            true_trajectory.append(self.true_params[0][0])

            est_trajectory.append(self.updater.est_mean()[0])

            if verbose: print("New eval ", self.updater.est_mean()[0])
            new_loss = eval_loss(self.model, self.updater.est_mean(), self.true_params)
            track_loss[idx_experiment] = new_loss[0]
            if verbose: print("New loss: ", new_loss[0])
            

            track_cov[idx_experiment] = np.sqrt( self.updater.est_covariance_mtx() )                
            if verbose: print("New cov: ", track_cov[idx_experiment])
            
        self.lastexp = self.lastexp + n_experiments
        
        if TBC:
            
            if verbose: print("True param = ", self.true_params )
            if verbose: print("Exparam = ", self.expparams[0][0] )
            
                
        if verbose: print('\n Tracking finished with reading = ' + str(self.updater.est_mean()[0]) )
        if verbose: print('##########################\n' )
        

        return [np.array(est_trajectory), track_loss, track_cov, track_acctime, track_pgh_time, track_time, np.array(true_trajectory)]



    
class DataPrecModel():
    r"""
    
    """
    
    ## INITIALIZER ##

    def __init__(self, freq_min=0.0, freq_max=1.0, n_particles=1000, wDeco = "Absent", mean_T = 0., var_T = None):
        
        if wDeco is "ToLearn":
            self.model = ExpDecoPrecessionModel(min_freq = freq_min)
        elif wDeco is "Known":
            self.model = ExpDecoKnownPrecessionModel(min_freq = freq_min, invT2 = mean_T)
        else:
            self.model = ExpPrecessionModel(min_freq = freq_min)
            
            
        self.n_particles = n_particles
        
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        self.fft_est = None
        self.bay_est = None
        
        self.wDeco = wDeco
        
        self.mean_T = mean_T
        self.var_T = var_T
     
    def classfft_prec(self, data):
    
        [ft_freq, spectrum, est_omega, xlimit, err_omega, norm] = fft_prec(data)

        self.fft_est = est_omega

        return [ft_freq, spectrum, self.fft_est]
    
    

    def est_prec(self, data, n_shots = 1, n_experiments = 50, resample_a=None, resample_thresh=0.5, verbose=False, TBC=False):
        """
        Class for (decohered) Ramsey fringe learning
        
        data: set of [time, likelihood] data, or "None" if you want simulation
        n_experiments: number of experiments to be performed before interruption
        resample_a: provides an indicator of how much the Liu-West resampler is allowed to "move" the distribution in case of resampling
            >> higher "a" indicatively leads to slower learning, but smoother learning
        resample_thresh: provides an indicator of how frequently to call resampling. 
            >> higher the "thresh" value, more frequently resampling events will occur, indicatively leading to faster but less smooth learning
        TBC: continues the learning from a previous prior instead of wiping it and restarting
        """
        
        if TBC:
            print("Continuing from...")
            print(self.prior)
        
        else:
            if self.wDeco is "ToLearn":
                self.prior=qi.ProductDistribution(qi.UniformDistribution([self.freq_min, self.freq_max]),qi.NormalDistribution(self.mean_T, self.var_T))
                self.updater = qi.SMCUpdater(self.model, self.n_particles, self.prior, resample_a=resample_a, resample_thresh=resample_thresh)
                self.heuristic = multiPGH(self.updater, inv_field=['w_', 'T_'], t_field='ts')
            else:
                self.prior = qi.UniformDistribution([self.freq_min, self.freq_max])
                self.updater = qi.SMCUpdater(self.model, self.n_particles, self.prior, resample_a=resample_a, resample_thresh=resample_thresh)
                self.heuristic = stdPGH(self.updater, inv_field='w_')          
            
            
        
        if data is None:
            self.sim = True
            if verbose: print("Simulated run with")
            true_params = self.prior.sample()
            
        elif len(data) is 1:
            self.sim = True
            if verbose: print("Simulated run with")
            true_params = data
            
        else:
            self.sim = False
            if verbose: print("Experimental run with")
            if self.fft_est != None:
                if self.wDeco is "ToLearn":
                    true_params = np.array([[self.fft_est, self.mean_T]])
                else:
                    true_params = np.array([[self.fft_est]])
                    
                    
            else:
                if self.wDeco is "ToLearn":
                    true_params = np.array([[self.classfft_prec(data)[2], self.mean_T ]])
                else:
                    true_params = np.array([[self.classfft_prec(data)[2]]])

                    if ( true_params[0][0] >= self.prior._ranges[0][1] or true_params[0][0] <= self.prior._ranges[0][0] ):
                        warnings.warn("Chosen prior appears incompatible with FFT analysis, consider modifying the range of the prior distribution")
            
            mydata = adjust_data_qi(data, n_shots = n_shots, omegabounds = [self.freq_min, self.freq_max])
        
        if verbose: print("(estimated) value: ", true_params)
        
        track_eval = []
        track_loss = np.empty(n_experiments)
        track_cov = np.empty(n_experiments)
        track_time = np.empty(n_experiments)
        track_pgh_time = np.empty(n_experiments)
        track_acctime = np.zeros(n_experiments)

        for idx_experiment in range(n_experiments):
            experiment = self.heuristic()
            track_pgh_time[idx_experiment] = deepcopy(experiment[0][0])
            
            if verbose: print("Proposed experiment ", experiment)
            
            if self.sim:
                datum = self.model.simulate_experiment(true_params, experiment)
            else:
                [datum, newtime] = retrieve_experiment(mydata, experiment)
                track_time[idx_experiment] = newtime
                experiment[0][0] = newtime
                if verbose: print("Found experiment ", experiment)
            
            track_acctime[idx_experiment] = track_acctime[max(0, idx_experiment-1)] + experiment[0][0]
            
            if verbose: print("Datum", datum)
            
            self.updater.update(datum, experiment)

            if self.wDeco is "ToLearn":
                track_eval.append(self.updater.est_mean())
            else:
                track_eval.append(self.updater.est_mean()[0])
            if verbose: print("New eval ", track_eval[idx_experiment])
            new_loss = eval_loss(self.model, self.updater.est_mean(), true_params)
            track_loss[idx_experiment] = new_loss[0]
            if verbose: print("New loss: ", new_loss[0])

            if self.wDeco is "ToLearn": 
                Tmax = self.mean_T+ 3*10*self.var_T  # self.mean_T # 
                fmax = self.freq_max
                adim = np.array(  [  [fmax**2, fmax*Tmax]  ,  [fmax*Tmax, Tmax**2]  ] ) 
                # adim = np.array(  [  [1,1]  ,  [1,1]  ] ) 
                track_cov[idx_experiment] = np.linalg.norm(  self.updater.est_covariance_mtx() / adim )
                # track_cov[idx_experiment] = np.linalg.norm(  self.updater.est_covariance_mtx() )
            else:
                track_cov[idx_experiment] = np.sqrt( self.updater.est_covariance_mtx() )
                
            if verbose: print("New cov: ", track_cov[idx_experiment])

        if verbose: print('\nFinal estimate is: ' + str(self.updater.est_mean()[0]) )
        if verbose: print('##########################\n' )
        
        if self.wDeco is "ToLearn":
            return [np.array(track_eval), track_loss, track_cov, track_acctime, track_pgh_time, track_time, 
                    self.updater.est_covariance_mtx() ]
        else:
            return [np.array(track_eval), track_loss, track_cov, track_acctime, track_pgh_time, track_time]
        
        
        
        
        
        
    def est_stepwise(self, data, chosen_ones, n_shots = 1, n_experiments = 50, resample_a=None, resample_thresh=0.5, wait_resample = 10, wait_cata = 50, brake_sigma = 10**0, verbose=False, 
    basic_verbose = True, TBC=False):
        """
        Class for learning stepwise changes in the frequency, with underlying Ramsey process
        
        data: set of [time, list_of_likelihoods] data, or "None" if you want simulation
            >> beware DIFFERENT SHAPE from est_prec method !!!
        n_experiments: number of experiments to be performed before interruption
        resample_a: provides an indicator of how much the Liu-West resampler is allowed to "move" the distribution in case of resampling
            >> higher "a" indicatively leads to slower learning, but smoother learning
        resample_thresh: provides an indicator of how frequently to call resampling. 
            >> higher the "thresh" value, more frequently resampling events will occur, indicatively leading to faster but less smooth learning
        TBC: continues the learning from a previous prior instead of wiping it and restarting
        
        wait_resample: threshold BELOW which the rate of resampling is considered suspicious
            >> intuitively, this corresponds to many particles having lost any significative weight
        wait_cata: threshold ABOVE which learning is considered stable, 
            >> and therefore excessive resampling is suspicious
            >> should be smaller than expected transition rate in the underlying Poissonian process
        brake_sigma: threshold to stop further learning from occurring, to allow for faster recovery 
            >> the lower it is, the more accurate the estimate when magnetic field is stable
            >> but the less likely and prompt MFL will recover from stepwise changes, especially if dramatic
        
        """
        
        ### TODO: SIMULATION not tested, probably must be debugged!!!!
        
        if TBC:
            print("Continuing from...")
            print(self.prior)
        
        else:
            self.prior = qi.UniformDistribution([self.freq_min, self.freq_max])
            self.updater = qi.SMCUpdater(self.model, self.n_particles, self.prior, resample_a=resample_a, resample_thresh=resample_thresh)
            self.heuristic = stdPGH(self.updater, inv_field='w_')          
            
            
        
        if data is None:
            self.sim = True
            if verbose: print("Simulated run with")
            true_params = self.prior.sample()
            
        elif len(data) is 1:
            self.sim = True
            if verbose: print("Simulated run with")
            true_params = data
            
        else:
            self.sim = False
            if verbose: print("Experimental run")

        
        track_eval = []
        track_loss = np.empty(n_experiments)
        track_cov = np.empty(n_experiments)
        track_time = np.empty(n_experiments)
        track_pgh_time = np.empty(n_experiments)
        track_acctime = np.zeros(n_experiments)

        resample_events = []
        potcata_events = []
        track_ness = []
        cata_events = [0]
        delta_resample = n_experiments
        delta_cata = n_experiments
        check_cata = True

        track_ess = []

        
        for idx_experiment in range(n_experiments):
            
            
            # HEURISTIC section
            experiment = self.heuristic()
        #     experiment[0][0] = slowme*experiment[0][0]    
                # slowing down the PGH (works to prevent excessive confidence in estimate, but slows down recovery too much)
        #     experiment[0][0] = data[0][idx_experiment]
                # progressive selection of times instead of PGH generated like in the diffusive learning (fails tragically for stepwise)
                
            track_pgh_time[idx_experiment] = deepcopy(experiment[0][0])

            if verbose: print("Proposed experiment ", experiment)
            
            
            # EXPERIMENT CHOICE section

            idx_data = int(chosen_ones[:,0][idx_experiment])
                
            mydata = adjust_data_qi([data[0], data[1][idx_data]], n_shots = 1, omegabounds = [self.freq_min, self.freq_max])
            
            if self.sim:
                datum = self.model.simulate_experiment(true_params, experiment)
            else:
                [datum, newtime] = retrieve_experiment(mydata, experiment)
                track_time[idx_experiment] = newtime
                experiment[0][0] = newtime
                if verbose: print("Found experiment ", experiment)

            track_acctime[idx_experiment] = track_acctime[max(0, idx_experiment-1)] + experiment[0][0]
            if verbose: print("Datum", datum)

            
            # UPDATES section
            self.updater.update(datum, experiment)
        #     self.updater.batch_update(np.array([datum]), np.array([experiment]), resample_interval= n_experiments)
            
            ##############################################################
            ### catastrophic event detection (and possible mending)    
            if self.updater.just_resampled: 
                resample_events.append(idx_experiment)
                check_cata = True

            if len(resample_events)>1 :
                delta_resample = resample_events[-1] - resample_events[-2]       
            
            if (delta_resample < wait_resample) and check_cata:
                if basic_verbose: print("Potential catastrophe at experiment ", idx_experiment)
                potcata_events.append(idx_experiment)
                check_cata = False
                delta_cata = idx_experiment - cata_events[-1]
                if (delta_cata > wait_cata):
                    cata_events.append(idx_experiment)
                    if verbose: print("!!! Catastrophe occurred at experiment ", idx_experiment)
                    self.prior = qi.UniformDistribution([self.freq_min, self.freq_max])
                    self.updater = qi.SMCUpdater(self.model, self.n_particles, self.prior, resample_a=resample_a, resample_thresh=resample_thresh)
                    self.heuristic = stdPGH(self.updater, inv_field='w_')
            ##############################################################

            # STORING section
            track_eval.append(self.updater.est_mean()[0])
            if verbose: print("New eval ", track_eval[idx_experiment])
            
            true_params = np.array([[  chosen_ones[:,1][idx_experiment]  ]])
            new_loss = eval_loss(self.model, self.updater.est_mean(), true_params)
            track_loss[idx_experiment] = new_loss[0]
            
            if verbose: print("New loss: ", new_loss[0])

            
            track_cov[idx_experiment] = np.sqrt( self.updater.est_covariance_mtx() )
            track_ness.append( self.updater.n_ess )
            
            ##############################################################################
            # try to prevent cov to shrink too much and make it unfeasible to detect changes because of lack of support
            if track_cov[idx_experiment] < brake_sigma:
        #         print("Excessive shrinking at experiment {:d}, moving backward".format(idx_experiment)  )
                self.prior = qi.NormalDistribution( mean = track_eval[-2], var = track_cov[idx_experiment-1] )
                self.updater = qi.SMCUpdater(self.model, self.n_particles, self.prior, resample_a=resample_a, resample_thresh=resample_thresh)
                self.heuristic = stdPGH(self.updater, inv_field='w_')    
            ##############################################################################

            if verbose: print("New cov: ", track_cov[idx_experiment])

        if basic_verbose: print("{:d} total resampling events occurred".format(len(resample_events))   )
        if basic_verbose: print("{:d} total catatrophic events occurred".format(len(cata_events))   )

        
        
        return [np.array(track_eval), track_loss, track_cov, 
                track_acctime, track_pgh_time, track_time, np.array(track_ness),
                np.array(resample_events), np.array(potcata_events), np.array(cata_events)]
                






                
class NoisyDataPrecModel():
    r"""
    
    """
    
    ## INITIALIZER ##

    def __init__(self, freq_min=0.0, freq_max=1.0, n_particles=1000, noise = "Absent", eta = 1.0):
        
        base_model = ExpPrecessionModel(min_freq = freq_min)
        
        if noise is "Absent":
            self.model = base_model
        elif noise is "Binomial":
            self.model = qi.BinomialModel(base_model)
        elif noise is "Unbalanced":
            self.model = qi.BinomialModel(AsymmetricLossModel(base_model, eta=eta))
            
        self.n_particles = n_particles
        
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        self.fft_est = None
        self.bay_est = None

     
    def classfft_prec(self, data):
    
        [ft_freq, spectrum, est_omega, xlimit, err_omega, norm] = fft_prec(data)

        self.fft_est = est_omega

        return [ft_freq, spectrum, self.fft_est]
    
    

    def est_prec(self, data, n_shots = 1, n_experiments = 50, resample_a=None, resample_thresh=0.5, verbose=False, TBC=False, use_heuristic = "PGH", heuristic_ratio = 1., heuristic_root = 1):
        """
        Class for (decohered) Ramsey fringe learning
        
        data: set of [time, likelihood] data, or "None" if you want simulation
        n_experiments: number of experiments to be performed before interruption
        resample_a: provides an indicator of how much the Liu-West resampler is allowed to "move" the distribution in case of resampling
            >> higher "a" indicatively leads to slower learning, but smoother learning
        resample_thresh: provides an indicator of how frequently to call resampling. 
            >> higher the "thresh" value, more frequently resampling events will occur, indicatively leading to faster but less smooth learning
        TBC: continues the learning from a previous prior instead of wiping it and restarting
        """
        
        if TBC:
            print("Continuing from...")
            print(self.prior)
        
        else:
            self.prior = qi.UniformDistribution([self.freq_min, self.freq_max])
            self.updater = qi.SMCUpdater(self.model, self.n_particles, self.prior, resample_a=resample_a, resample_thresh=resample_thresh)
            if use_heuristic is "PGH":
                self.heuristic = stdPGH(self.updater, inv_field='w_')       
            elif use_heuristic is "rootPGH":
                self.heuristic = rootPGH(self.updater, root = 1, inv_field='w_')                       
            else:
                self.heuristic = stepwise_heuristic(self.updater, n_shots=1, ts = data[0])
          
        
        if data is None:
            self.sim = True
            if verbose: print("Simulated run with")
            true_params = self.prior.sample()
            
        elif len(data) is 1:
            self.sim = True
            if verbose: print("Simulated run with")
            true_params = data
            
        else:
            self.sim = False
            if verbose: print("Experimental run with")
            if self.fft_est != None:
                true_params = np.array([[self.fft_est]])                                
            else:
                true_params = np.array([[self.classfft_prec(data)[2]]])

                if ( true_params[0][0] >= self.prior._ranges[0][1] or true_params[0][0] <= self.prior._ranges[0][0] ):
                    warnings.warn("Chosen prior appears incompatible with FFT analysis, consider modifying the range of the prior distribution")
            
            
            mydata = adjust_data_qi(data, n_shots = n_shots, omegabounds = [self.freq_min, self.freq_max])
        
        if verbose: print("(estimated) value: ", true_params)
        
        track_eval = []
        track_loss = np.empty(n_experiments)
        track_cov = np.empty(n_experiments)
        track_time = np.empty(n_experiments)
        track_pgh_time = np.empty(n_experiments)
        track_acctime = np.zeros(n_experiments)

        for idx_experiment in range(n_experiments):
            experiment = self.heuristic()
            experiment[0][0] = heuristic_ratio*experiment[0][0]
            track_pgh_time[idx_experiment] = deepcopy(experiment[0][0])
            
            if verbose: print("Proposed experiment ", experiment)
            
            if self.sim:
                datum = self.model.simulate_experiment(true_params, experiment)
            else:
                [datum, newtime] = retrieve_experiment(mydata, experiment)
                track_time[idx_experiment] = newtime
                experiment[0][0] = newtime
                if verbose: print("Found experiment ", experiment)
            
            track_acctime[idx_experiment] = track_acctime[max(0, idx_experiment-1)] + experiment[0][0]
            
            if verbose: print("Datum", datum)
            
            self.updater.update(datum, experiment)


            track_eval.append(self.updater.est_mean()[0])
            if verbose: print("New eval ", track_eval[idx_experiment])
            new_loss = eval_loss(self.model, self.updater.est_mean(), true_params)
            track_loss[idx_experiment] = new_loss[0]
            if verbose: print("New loss: ", new_loss[0])
            
            track_cov[idx_experiment] = np.sqrt( self.updater.est_covariance_mtx() )  
            if verbose: print("New cov: ", track_cov[idx_experiment])

        if verbose: print('\nFinal estimate is: ' + str(self.updater.est_mean()[0]) )
        if verbose: print('##########################\n' )
        

        return [np.array(track_eval), track_loss, track_cov, track_acctime, track_pgh_time, track_time]

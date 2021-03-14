"""
A module for a mixture density network layer
(_Mixture Desity Networks_ by Bishop, 1994.)
"""
import sys
import torch
import torch.tensor as ts
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import math

# Draw distributions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

'''
Process:
    Input:x -> Some model (body) -> Characteristic vector:z (feature)
            -> MDN (head)        -> Probabilistic  vector:p (output)
'''

class MDN_Module(nn.Module):
    """ 
    A Mixture Density Network Module
    
    Symbols: 
        B - Batch size
        G - Number of Gaussian components
        D - Input's dimensions
        F - Feature's dimensions
        C - Output's dimensions (Gaussian distribution's dimensions)
    
    Arguments:
        dim_fea  (int): the feature's dimensions
        dim_prob (int): the output's dimenssions
        num_gaus (int): the number of Gaussians per output dimension
    
    Input:
        minibatch (BxF)
    Output:
        (alp, mu, sigma) (BxG, BxGxC, BxGxC)
        alp   - (alpha) Component's weight
        mu    - Mean value
        sigma - Standard deviation
    """
    def __init__(self, dim_fea, dim_prob, num_gaus):
        super(MDN_Module, self).__init__()
        self.dim_fea = dim_fea
        self.dim_prob = dim_prob
        self.num_gaus = num_gaus
        self.layer_alp = nn.Sequential(
            nn.Linear(dim_fea, num_gaus),
            nn.Softmax(dim=1) # If 1, go along each row
        )
        self.layer_mu    = nn.Linear(dim_fea, dim_prob*num_gaus)
        self.layer_sigma = nn.Sequential(
            nn.Linear(dim_fea, dim_prob*num_gaus),
            ReExp_Layer() 
        )

    def forward(self, batch):
        alp = self.layer_alp(batch)
        mu = self.layer_mu(batch)
        mu = mu.view(-1, self.num_gaus, self.dim_prob)
        sigma = self.layer_sigma(batch)
        sigma = sigma.view(-1, self.num_gaus, self.dim_prob)
        return alp, mu, sigma

class ReExp_Layer(nn.Module):
    """ 
    A modified exponential layer.
    Only the negative part of the exponential retains.
    The positive part is linear: y=x+1.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l = nn.ELU() # ELU: max(0,x)+min(0,α∗(exp(x)−1))
        return torch.add(l(x), 1) # assure no negative sigma produces!!!

class classic_MDN_Module(nn.Module):
    def __init__(self, dim_fea, dim_prob, num_gaus):
        super(classic_MDN_Module, self).__init__()
        self.dim_fea = dim_fea
        self.dim_prob = dim_prob
        self.num_gaus = num_gaus
        self.layer_alp = nn.Sequential(
            nn.Linear(dim_fea, num_gaus),
            nn.Softmax(dim=1) # If 1, go along each row
        )
        self.layer_mu    = nn.Linear(dim_fea, dim_prob*num_gaus)
        self.layer_sigma = nn.Sequential(
            nn.Linear(dim_fea, dim_prob*num_gaus)
        )

    def forward(self, batch):
        alp = self.layer_alp(batch)
        mu = self.layer_mu(batch)
        mu = mu.view(-1, self.num_gaus, self.dim_prob)
        sigma = torch.exp(self.layer_sigma(batch))
        sigma = sigma.view(-1, self.num_gaus, self.dim_prob)
        return alp, mu, sigma


def cal_GauProb(mu, sigma, x):
    """
    Return the probability of "data" given MoG parameters "mu" and "sigma".
    
    Arguments:
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
        x     (BxC)   - A batch of data points.

    Return:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding mu/sigma index.
    """
    x = x.unsqueeze(1).expand_as(mu) # BxC -> Bx1xC -> BxGxC
    prob = torch.rsqrt(torch.tensor(2*math.pi)) * torch.exp(-((x - mu) / sigma)**2 / 2) / sigma
    return torch.prod(prob, dim=2) # overall probability for all output's dimensions in each component, BxG

def cal_multiGauProb(alp, mu, sigma, x):
    """
    Arguments:
        alp   (BxG)   - (alpha) Component's weight
    """
    prob = alp * cal_GauProb(mu, sigma, x) # BxG
    prob = torch.sum(prob, dim=1) # Bx1
                                  # overall prob for each batch (sum is for all compos)
    return prob

def loss_NLL(alp, mu, sigma, data):
    """
    Calculates the error, given the MoG parameters and the data.
    The loss is the negative log likelihood of the data given the MoG parameters.
    """
    nll = -torch.log(cal_multiGauProb(alp, mu, sigma, data)) 
    return torch.mean(nll)

def loss_MaDist(alp, mu, sigma, data): # Mahalanobis distance
    '''
    mu    (GxC) - The means of the Gaussians. 
    sigma (GxC) - The standard deviation of the Gaussians.
    '''
    md = []
    alp = alp/sum(alp) #normalization
    for i in range(mu.shape[0]): # do through every component
        mu0 = (data-mu[i,:]).unsqueeze(0) # (x-mu)
        S_inv = ts([[1/sigma[i,0],0],[0,1/sigma[i,1]]]) # S^-1 inversed covariance matrix
        md0 = torch.sqrt( S_inv[0,0]*mu0[0,0]**2 + S_inv[1,1]*mu0[0,1]**2 )
        md.append(md0)
    return ts(md), sum(ts(md)*alp)

def loss_EMD(): pass

def sample(alp, mu, sigma):
    """
    Draw samples from a MoG.
    Return one sample for each batch
    """
    categorical = Categorical(alp) # aka. generalized Bernoulli
    try:
        alps = list(categorical.sample().data) # take a sample of alpha for each batch
    except:
        raise Exception('Ooooops! Model collapse!')
    sample = sigma.new_empty(sigma.size(0), sigma.size(2)).normal_() # sample of (0,1) normal distribution
    for i, idx in enumerate(alps):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample


def take_mainCompo(alp, mu, sigma, main=3):
    alp = alp[0,:]
    mu = mu[0,:,:]
    sigma = sigma[0,:,:]
    main_alp = alp[:main]       # placeholder
    main_mu = mu[:main,:]       # placeholder
    main_sigma = sigma[:main,:] # placeholder
    _, indices = torch.sort(alp) # ascending order
    for i in range(1,main+1):
        idx = indices[-i].item() # largest to smallest
        main_alp[i-1] = alp[idx]
        main_mu[i-1,:] = mu[idx,:]
        main_sigma[i-1,:] = sigma[idx,:]
    return main_alp.unsqueeze(0), main_mu.unsqueeze(0), main_sigma.unsqueeze(0) # insert the "batch" dimension

def take_goodCompo(alp, mu, sigma, thre=0.1):
    alp = alp[0,:]
    mu = mu[0,:,:]
    sigma = sigma[0,:,:]
    thre = thre*max(alp)
    idx = (alp>thre)
    good_alp = alp[idx]
    good_mu = mu[idx,:]
    good_sigma = sigma[idx,:]
    return good_alp.unsqueeze(0), good_mu.unsqueeze(0), good_sigma.unsqueeze(0) # insert the "batch" dimension

def sigma_limit(mu, sigma, nsigma=3):
    # nsigma: 1 -> 0.6827   2 -> 0.9545   3 -> 0.9974
    x_scope = [(mu-nsigma*sigma)[0,:,0], (mu+nsigma*sigma)[0,:,0]]
    y_scope = [(mu-nsigma*sigma)[0,:,1], (mu+nsigma*sigma)[0,:,1]]
    x_min = torch.min(x_scope[0])
    x_max = torch.max(x_scope[1])
    y_min = torch.min(y_scope[0])
    y_max = torch.max(y_scope[1])
    if x_min !=  torch.min(abs(x_scope[0])):
        x_min = -torch.min(abs(x_scope[0]))
    if x_max !=  torch.max(abs(x_scope[1])):
        x_max = -torch.max(abs(x_scope[1]))
    if y_min !=  torch.min(abs(y_scope[0])):
        y_min = -torch.min(abs(y_scope[0]))
    if y_max !=  torch.max(abs(y_scope[1])):
        y_max = -torch.max(abs(y_scope[1]))
    return [x_min, x_max], [y_min, y_max]

def cal_multiGauProbDistr(xx, yy, alp, mu, sigma):
    xy = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=1).astype(np.float32)
    p = np.array([])
    for i in range(xy.shape[0]):
        p = np.append( p, cal_multiGauProb(alp, mu, sigma, x=ts(xy[i,:][np.newaxis,:])).detach().numpy() )
    p[np.where(p<max(p)/10)] = 0
    return p.reshape(xx.shape)

def draw_probDistribution(ax, alp, mu, sigma, main=3, nsigma=3, step=0.5, colorbar=False, toplot=True):
    '''
    Arguments:
        ax            - Axis
        alp   (BxG)   - (alpha) Component's weight.
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
    '''
    if main is not None:
        alp, mu, sigma = take_mainCompo(alp, mu, sigma, main=main)

    # ================= Register Colormap ================START
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors)) # get colormap
    color_array[:,-1] = np.linspace(0,1,ncolors) # change alpha values
    color_array[:,-1][:25] = 0
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array) # create a colormap object
    plt.register_cmap(cmap=map_object) # register this new colormap with matplotlib
    # ================= Register Colormap ==================END

    xlim, ylim = sigma_limit(mu, sigma, nsigma=nsigma)
    x = np.arange(xlim[0].detach().numpy(), xlim[1].detach().numpy(), step=step)
    y = np.arange(ylim[0].detach().numpy(), ylim[1].detach().numpy(), step=step)
    xx, yy = np.meshgrid(x, y)

    pp = cal_multiGauProbDistr(xx, yy, alp, mu, sigma)

    if toplot:
        cntr = ax.contourf(xx, yy, pp, cmap="rainbow_alpha")
        if colorbar:
            plt.colorbar(cntr, ax=ax)

    return xx,yy,pp

def draw_GauEllipse(ax, mu, sigma, fc='b', nsigma=3, extend=False, label=None):
    '''
    mu    (GxC) - The means of the Gaussians. 
    sigma (GxC) - The standard deviation of the Gaussians.
    '''
    for i in range(mu.shape[0]):
        if i != 0:
            label=None
        if extend:
            patch = patches.Ellipse(mu[i,:], nsigma*sigma[i,0]+8, nsigma*sigma[i,1]+8, fc=fc, label=label)
            ax.add_patch(patch)
        else:
            patch = patches.Ellipse(mu[i,:], nsigma*sigma[i,0], nsigma*sigma[i,1], fc=fc, label=label)
            ax.add_patch(patch)
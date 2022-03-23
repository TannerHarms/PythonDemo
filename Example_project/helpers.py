'''
A set of functions to help get the CylinderPOD.
'''

# General
import numpy as np
import time

# File management
import urllib.request as urllib
import zipfile
import scipy.io as sio
import os
import pickle

# Plotting functionality
import matplotlib.pyplot as plt


def GetCylinderFlowData():
    
    # Get relative path information
    dirname = os.path.dirname(__file__)
    datafldr = os.path.join(dirname, 'data')
    if not os.path.exists(datafldr):    # Make the directory if not exists
        os.makedirs(datafldr)
    datafile = os.path.join(datafldr, 'Cyl_Data.pkl')
    
    # Check if data is saved locally
    if os.path.exists(datafile):
        with open(datafile, 'rb') as handle:
            data = pickle.load(handle)
        
    else:   # download the data from the web  
        print('Downloading data from DMDbook.com')
        # URL location of the data
        url = 'http://dmdbook.com/DATA.zip'
        
        # Scrape the data from the website
        start_time = time.time()
        filehandle, _ = urllib.urlretrieve(url)
        zipfile_object = zipfile.ZipFile(filehandle, 'r')
        print(f'time to download: {time.time()-start_time}')
        
        # Create an object for the data
        cyl_data_file = zipfile_object.open('DATA/FLUIDS/CYLINDER_ALL.mat')

        # cyl_data_file = r'C:\Users\harms\OneDrive\Desktop\PythonDemo\Example_project\data\CYLINDER_ALL.mat'
        mat = sio.loadmat(cyl_data_file)
        print(sorted(mat.keys()))       # What are the contents of the file?
        
        # Store as a new dictionary
        data = {
            'U'     : mat['UALL'],
            'V'     : mat['VALL'],
            'Vort'  : mat['VORTALL'],
            'm'     : mat['m'],
            'n'     : mat['n'],
            'nx'    : mat['nx'],
            'ny'    : mat['ny'],
        }

        # save data as an pkl file
        with open(datafile, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('data fields: ', data.keys())
    return data

def SaveFig(figname):
    # Saves a figure to the FIGS folder in the document section.
    
    dirname = os.path.dirname(__file__)
    figfldr = os.path.join(dirname, 'document', 'FIGS')
    if not os.path.exists(figfldr):    # Make the directory if not exists
        os.makedirs(figfldr)
    figfile = os.path.join(figfldr, figname + '.eps')
    
    plt.savefig(figfile, format='eps', dpi=200)

class POD:
    def __init__(self, data, field='Vort', centered=True):
        
        # Set up class properties
        if not (field=='Vort' or field=='U' or field=='V'):
            field = 'Vort'
        self.X = np.matrix(data[field])
        if centered:
            self.meanX = np.mean(self.X, axis=1)
            self.X -= self.meanX
        self.m = int(data['m'])
        self.n = int(data['n'])
        self.nx = int(data['nx'])
        self.ny = int(data['ny'])
    
    def fit(self, method='classic'):
        
        self.method = method
        if method=='snapshot':
            self.snapshot()
        elif method=='svd':
            self.svd()
        else:
            self.classic()       
        
    def classic(self):
        K = self.X*self.X.T
        
        # Compute Eigenvalue problem
        Lambda, self.Phi = np.linalg.eig(K)
        
        # Sort from largest to smallest
        idx = Lambda.argsort()[::-1]   
        Lambda = Lambda[idx];        self.Phi = self.Phi[:,idx]
        
        # Get the singular values
        self.Sigma = np.sqrt(Lambda)
        
        # Time varying coefficients:
        self.Psi = self.X.T*self.Phi*np.diag(np.reciprocal(self.Sigma))
            
    def snapshot(self):
        K = self.X.T*self.X
        
        # Compute Eigenvalue problem
        Lambda, self.Psi = np.linalg.eig(K)
        
        # Sort from largest to smallest
        idx = Lambda.argsort()[::-1]   
        Lambda = Lambda[idx];        self.Psi = self.Psi[:,idx]
        
        # Get the singular values
        self.Sigma = np.sqrt(Lambda)
        
        # Modes:
        self.Phi = self.X*self.Psi*np.diag(np.reciprocal(self.Sigma))
        
    def svd(self):
        self.Phi, self.Sigma, self.Psi = np.linalg.svd(self.X, full_matrices=False)

class PODplot:
    
    def __init__(self, PODobj):
        self.pod = PODobj
        
    def reconstruct(self, mode_idx):
        
        # Initialize the approximation
        Xhat = np.zeros_like(self.pod.X)
        
        # Sum the modes in the mode_idx additively.
        for i in mode_idx:
            Xhat += self.pod.Phi[:,i]*self.pod.Sigma[i]*self.pod.Psi[:,i].T 
        
        # reshape the matrix a meaningful perspective.
        Xhat_reshape = []
        for i in range(np.shape(self.pod.X)[1]):
            Xhat_reshape.append(np.reshape(Xhat[:,i], (self.pod.ny, self.pod.nx)) )
        
        return np.transpose(Xhat_reshape, (0, 2, 1))
        
    def energy(self, **kwargs):
        
        # relative energy
        relen = self.pod.Sigma / np.sum(self.pod.Sigma)
        plt.plot(1-relen)
        plt.xlim([0,25])
        plt.grid()
    
    def modes(self, mode_idx, **kwargs):
        
        # Reconstruct the mode
        Xhat = self.reconstruct(mode_idx)
        plt.pcolormesh(Xhat[0], linewidth=0, rasterized=True)
        plt.axis('scaled')
        plt.set_cmap('coolwarm')
    
    def tvc(self, mode_idx, **kwargs):
        
        for i in mode_idx:
            plt.plot(self.pod.Sigma[i]*self.pod.Psi[:,i], label='a = %s' % i)
            
        plt.legend()
        plt.grid()
        

# An entry point for debugging this code.  
if __name__ == '__main__': 
    pass

    
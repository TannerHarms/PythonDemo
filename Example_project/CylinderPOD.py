'''
Entry point for the Python Demo Example:

Author: Tanner Harms
Date: 03/23/2022
'''

# Preamble:  Import the necessary packages, etc.
import numpy as np
import os

# These are self-made functions from another file called helpers.  
# * means import native naming, so I dont have to say helpers.(function).
from helpers import *

# Body of the script:
# Import the data from DMDbook.com or from your file structure
data = GetCylinderFlowData()    

# Perform a POD of the data using the snapshot method
pod = POD(data)
pod.fit(method='snapshot')

# look at the size of the output
print(f'Modes are of shape {np.shape(pod.Phi)}')
print(f'Energies are of shape {np.shape(pod.Sigma)}')
print(f'Time-varying coefficients are of shape {np.shape(pod.Psi)}')

# Make a plotting object.
podplt = PODplot(pod)

# Plot and save the Energy spectrum
plt.figure(); podplt.energy(); SaveFig('Cumulative_Energy')
plt.show()

# plot and save the first 5 modes
plt.figure(); podplt.modes([0]); SaveFig('Mode0')
plt.figure(); podplt.modes([1]); SaveFig('Mode1')
plt.figure(); podplt.modes([2]); SaveFig('Mode2')
plt.figure(); podplt.modes([3]); SaveFig('Mode3')
plt.figure(); podplt.modes([4]); SaveFig('Mode4')
plt.show()

# Plot and save the time varying coefficients
plt.figure(); podplt.tvc([0,1,2,3,4]); SaveFig('TVC_1through5')
plt.show()

import os
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage as nd
from skimage.transform import rescale, resize
import astropy.io.fits as pyfits
from itertools import combinations
import random
to_rd = lambda m, d: m * np.exp(1j * np.deg2rad(d))
to_pd = lambda x: (abs(x), np.rad2deg(np.angle(x)))

def data_reader(inobs):

	# Load the uvfits file
        obs = eh.obsdata.load_uvfits(inobs)

        # Scan-average the data
        # Identify the scans (times of continous observation) in the data
        obs.add_scans()

        # Coherently average the scans, which can be averaged due to ad-hoc phasing
        
        obs = obs.avg_coherent(1800., scan_avg=True)

        # Estimate the total flux density from the ALMA(AA) -- APEX(AP) zero baseline
        zbl = np.median(obs.unpack_bl('AA','AP','amp')['amp'])
        
        # Flag out sites in the obs.tarr table with no measurements
        allsites = set(obs.unpack(['t1'])['t1'])|set(obs.unpack(['t2'])['t2'])
        obs.tarr = obs.tarr[[o in allsites for o in obs.tarr['site']]]
        obs = eh.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obs.data, obs.tarr,
                         source=obs.source, mjd=obs.mjd,
                         ampcal=obs.ampcal, phasecal=obs.phasecal)

        obs = obs.flag_bl(['AA', 'AP'], output='kept')
        obs = obs.flag_bl(['JC', 'SM'], output='kept')

        obs.add_cphase(avg_time=150)
                
        ###################################WORKING HERE ######################3
        copy_data = obs.cphase.copy()
        
        inds = np.unique(obs.cphase['time'])
        
        index_cp = np.zeros([len(obs.cphase['time']), 4])
        ll = 0
        for i in range(len(inds)):
            inds_bl = np.where(obs.data['time'] == inds[i])
            
            if inds_bl[0].size == 0:
                print('No data')
            else:
                sta1 = obs.data['t1'][inds_bl]
                sta2 = obs.data['t2'][inds_bl]
            
                inds_cp = np.where(obs.cphase['time'] == inds[i])
                sta_cp1 = obs.cphase['t1'][inds_cp] 
                sta_cp2 = obs.cphase['t2'][inds_cp]
                sta_cp3 = obs.cphase['t3'][inds_cp]
        
                copy_data['time'][ll:ll + len(inds_cp[0])] = obs.cphase['time'][inds_cp[0]]
                copy_data['t1'][ll:ll + len(inds_cp[0])] = obs.cphase['t1'][inds_cp[0]]
                copy_data['t2'][ll:ll + len(inds_cp[0])] = obs.cphase['t2'][inds_cp[0]]
                copy_data['t3'][ll:ll + len(inds_cp[0])] = obs.cphase['t3'][inds_cp[0]]
                copy_data['u1'][ll:ll + len(inds_cp[0])] = obs.cphase['u1'][inds_cp[0]]
                copy_data['v1'][ll:ll + len(inds_cp[0])] = obs.cphase['v1'][inds_cp[0]]
                copy_data['u2'][ll:ll + len(inds_cp[0])] = obs.cphase['u2'][inds_cp[0]]
                copy_data['v2'][ll:ll + len(inds_cp[0])] = obs.cphase['v2'][inds_cp[0]]
                copy_data['u3'][ll:ll + len(inds_cp[0])] = obs.cphase['u3'][inds_cp[0]]
                copy_data['v3'][ll:ll + len(inds_cp[0])] = obs.cphase['v3'][inds_cp[0]]
                copy_data['cphase'][ll:ll + len(inds_cp[0])] = obs.cphase['cphase'][inds_cp[0]]
                copy_data['sigmacp'][ll:ll + len(inds_cp[0])] = obs.cphase['sigmacp'][inds_cp[0]]
        
                for k in range(len(inds_cp[0])):
                
                    ind1 = np.where((sta1 == sta_cp1[k]) & (sta2 == sta_cp2[k]))
                    ind2 = np.where((sta1 == sta_cp2[k]) & (sta2 == sta_cp3[k]))
                    ind3 = np.where((sta1 == sta_cp1[k]) & (sta2 == sta_cp3[k]))            
                    index_cp[ll+k, 0] = inds[i]
                    index_cp[ll+k, 1] = inds_bl[0][ind1[0]][0]
                    index_cp[ll+k, 2] = inds_bl[0][ind2[0]][0]
                    index_cp[ll+k, 3] = inds_bl[0][ind3[0]][0]
                ll = ll + len(inds_cp[0])
        
        #fig, ax1 = plt.subplots(1,1)
        #ax1.plot(np.sqrt(obs.data['u']**2 + obs.data['v']**2), np.abs(obs.data['vis']), 'or')
        #plt.show()
        #pdb.set_trace()
        
        data_dict = {'u':obs.data['u'], 'v':obs.data['v'], 'vis':np.abs(obs.data['vis']), 'vsigma':obs.data['vsigma'], 'cphase':copy_data['cphase'],\
        'sigmacp':copy_data['sigmacp'], 'index_cp':index_cp, 'u1':copy_data['u1'], 'v1':copy_data['v1'], 'u2':copy_data['u2'], 'v2':copy_data['v2'] }
        
        
        return data_dict
                
                
                
                

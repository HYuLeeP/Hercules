# Import stuff

import astropy.io.fits as ft
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as cl
from matplotlib.path import Path
import pickle

###############################################################################
# identify if a star is hercules
def if_hercules(idx_star,dataset,Lz='L_Z',Vr='vR_Rzphi'):
    '''
    Input:
        idx_star: the index of data in the dataset
        dataset: a np.field data set containing Lz and 'vR_Rzphi' entries
    '''
    if dataset[idx_star][Lz]<2.5*dataset[idx_star][Vr]+1720:
        if dataset[idx_star][Lz]<-20*dataset[idx_star][Vr]+3400:
            if dataset[idx_star][Lz]>2.5*dataset[idx_star][Vr]+1200:
                if dataset[idx_star][Lz]>-5*dataset[idx_star][Vr]+1250:
                    return True
    return False

###############################################################################
# get %of hercules
def portion_hercules(star_set_idx,ndynamics):
    n=len(star_set_idx)
    m=0
    for i in tqdm(range(n)):
        idx=star_set_idx[i]
        pt=(ndynamics['vR_Rzphi'][idx],ndynamics['L_Z'][idx])
        if is_hercules(pt):
            m+=1
    return m/n
###############################################################################
# Splitting neighbour_star_data by [Fe/H] Abundances
# Take intervals: (~-0.5) (-0.5,-0.3) (-.3,-.1) (-.1,.1) (.1, .3) (.3,~)
# intersection=0
def split(Fe_list,intersection):
    Interval=[[-10,intersection],[intersection,10]]
    cluster_idx=[]
    for i in range(len(Interval)):
        bounds=Interval[i]
        lb=bounds[0]
        ub=bounds[1]
        # print(lb,ub)
        globals()[str(lb)+'to'+str(ub)+'_index']=[]
        for i in range(len(Fe_list)):
            if lb < Fe_list[i] and Fe_list[i] < ub:
                globals()[str(lb)+'to'+str(ub)+'_index'].append(i)
        # print(len(globals()[str(lb)+'to'+str(ub)+'_index']))
        cluster_idx.append(globals()[str(lb)+'to'+str(ub)+'_index'])

    return cluster_idx[0],cluster_idx[1]

###############################################################################
#note that if the star has NaN in 'vR_Rzphi' or 'L_Z', it is not included
def abundance(star_data,neighbour_star_data,chem_abun='fe_h'):
    return_list=[]
    n=int(0)
    for i in tqdm(range(len(star_data))):
        if star_data['sobject_id'][i] == neighbour_star_data['sobject_id'][n]:
            return_list.append(star_data[chem_abun][i])
            n+=1
    return return_list
###############################################################################
# eliminating statistical error by randomly truncating exceesive data to normalise number of stars in the bin
def norm_len(Interval,input_list):
    '''
    Input:
        Interval: list of upper and lower bounds
                e.g [[-10,-0.5],[-0.5,-0.3],[-.3,-.1],[-.1,0],[0, .1],[.1,10]]
                Note: global variables has been created already by these intervals.
    Output:
        list of lists of idx_list for intervals
    '''
    
    len_list=np.zeros(len(Interval))
    for i in range(len(Interval)):
        # bounds=Interval[i]
        # lb=bounds[0]
        # ub=bounds[1]
        temp=input_list[i]
        len_list[i]=len(temp)
    min_len=min(len_list)
    min_idx=np.argmin(len_list)
    return_list=[]
    for i in range(len(Interval)):
        # print(i)
        # bounds=Interval[i]
        # lb=bounds[0]
        # ub=bounds[1]
        temp_list=input_list[i]
        if i == min_idx:
            return_list.append(temp_list)
            continue
        remov_list=np.sort(np.random.randint(int(len_list[i])-1, size=(int(len_list[i]-min_len))))
        n=0
        for j in remov_list:

            temp_list.pop(j-n)
            n+=1
        return_list.append(temp_list)

    return return_list
###############################################################################
with open("hercules_contour_paths.pkl", "rb") as f:
    paths = pickle.load(f)

def is_hercules(pt):
    is_inside = any(path.contains_point(pt) for path in paths)
    return is_inside
###############################################################################

def nanfrac(alist):
    return len(alist[~np.isnan(alist)])/len(alist)
###############################################################################

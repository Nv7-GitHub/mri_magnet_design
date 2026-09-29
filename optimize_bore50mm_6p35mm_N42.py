# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:03:57 2023

@author: leeor alon
"""

import magsimulator
import magcadexporter
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from magpylib.magnet import Cuboid, CylinderSegment
import itertools
from scipy.spatial.transform import Rotation as R
import pandas as pd
import cProfile
import sys
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, freeze_support
from os import getpid
import time
import addcopyfighandler
import pygad
import numpy.matlib
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.operators.crossover.hux import HalfUniformCrossover

from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool
import pickle

import warnings
warnings.filterwarnings("ignore")

# usage: python optimize_bore50mm_6p35mm_N42.py <max magnets> <generations>
MAX_MAGNETS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 300

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

# from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize

class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        
        # variables = dict()
        # variables["layers"] = Integer(bounds=(1, 5))
        # variables["zpositions"] = Integer(bounds=(2, 20))
        # variables["mags_zero_pos"]=Integer(bounds=(np.array([8,8,8,8,8]), np.array([24,24,24,24,24])))
        # variables["increment_z_positions"]=Real(bounds=(np.sqrt(3)*12.7*np.ones((100,1)), np.sqrt(3)*12.7*4*np.ones((100,1))))
        # variables["number_mags_per_ring"] = Integer(bounds=(8*np.ones((100,1)), 24*np.ones((100,1))))
        self.maxzpos = int(24)
        self.maxlayers = int(2)
        self.increment = int(self.maxzpos*self.maxlayers)
        
        self.cube_side_length = 6.35 # 1/4 inch N42 cube
        self.maxzextent = 1000
        self.maxnumberofmagnets = MAX_MAGNETS
        self.rmin=25+3+self.cube_side_length/np.sqrt(2) # 50 mm bore + 3 mm former wall + half cube diagonal = 32.49
        self.dr=np.sqrt(3)*self.cube_side_length+3
        
        # print(self.increment)
        variables = dict()

        variables["x01"] = Integer(bounds=(1, self.maxlayers))
        variables["x02"] = Integer(bounds=(2, self.maxzpos))
        
        for k in range(3, 8):
            variables[f"x{k:02}"] = Integer(bounds=(4, 17)) # center ring, limited by layer 1 (max 18 fit at rmin)
        
        for k in range(8, 8+self.increment):
            variables[f"x{k:02}"] = Real(bounds=(self.cube_side_length+3, np.sqrt(3)*self.cube_side_length*3))
            
        # layer 1 max 24 magnet per ring
        for k in range(8+self.increment, 8+self.increment+self.increment//self.maxlayers):
            variables[f"x{k:02}"] = Integer(bounds=(4, 17)) # layer 1, max 18 fit at r=32.5
            # variables[f"x{k:02}"] = Choice(options=[8, 12, 16])         
            
        
        # # layer 2 max 36 magnet per ring    
        for k in range(8+self.increment+self.increment//self.maxlayers, 8+self.increment+2*self.increment//self.maxlayers):
            variables[f"x{k:02}"] = Integer(bounds=(4, 25)) # layer 2, max 26 fit at r=46.5
            # variables[f"x{k:02}"] = Choice(options=[12, 16, 24])       


        # # layer 3 and on max 42 magnet per ring
        for k in range(8+self.increment+2*self.increment//self.maxlayers, 8+2*self.increment):
            variables[f"x{k:02}"] = Integer(bounds=(4, 42)) 
            # variables[f"x{k:02}"] = Choice(options=[16, 24, 36])            

        
        for k in range(8+self.increment*2, 8+self.increment*3):
            variables[f"x{k:02}"] = Binary()               
        
        for k in range(8+self.increment*3, 8+self.increment*4):
            variables[f"x{k:02}"] = Real(bounds=(0, np.pi/8)) 
            
        super().__init__(vars=variables,n_ieq_constr=1, n_obj=2, **kwargs)


    def _evaluate(self, x, out, *args, **kwargs):
        layers=np.array(x[f"x01"].reshape((1,1))).astype(int).flatten()
        layers=layers[0]
        zpositions=np.array(x[f"x02"]).reshape((1,1)).astype(int)
        zpositions=zpositions[0]
        mags_per_endring_zero_pos=np.array([x[f"x{k:02}"] for k in range(3, 8)]).reshape((5,1)).astype(int).flatten()
        increment_z_matrix=np.array([x[f"x{k:02}"] for k in range(8, 8+self.increment)]).reshape((self.increment,1)).flatten()
        num_mags_matrix=np.array([x[f"x{k:02}"] for k in range(8+self.increment, 8+self.increment*2)]).reshape((self.increment,1)).astype(int).flatten()
        placement_ring_decision=np.array([x[f"x{k:02}"] for k in range(8+self.increment*2, 8+self.increment*3)]).reshape((self.increment,1)).astype(int).flatten()
        phase_diff_mat=np.array([x[f"x{k:02}"] for k in range(8+self.increment*3, 8+self.increment*4)]).reshape((self.increment,1)).flatten()
        # print(layers.shape)
        # print(zpositions.shape)
        # print(mags_per_endring_zero_pos.shape)
        # print(increment_z_matrix.shape)
        # print(num_mags_matrix.shape)
        #constants
        
        increment_z_matrix=increment_z_matrix.reshape((self.maxlayers,self.maxzpos))
        num_mags_matrix=num_mags_matrix.reshape((self.maxlayers,self.maxzpos))
        placement_ring_decision = placement_ring_decision.reshape((self.maxlayers,self.maxzpos))
        phase_diff_mat = phase_diff_mat.reshape((self.maxlayers,self.maxzpos))
        
        if(np.mod(zpositions,2)==0):
            z_half = int(zpositions/2)
        else:
            z_half = int((zpositions-1)/2)
            
        point_list=[]
        # print(layers)
        # print(type(layers))
        for ii in range(layers):
            r_cur=self.rmin+ii*self.dr
            
            if(np.mod(zpositions,2)!=0):
                endring = magsimulator.generate_ring_of_magnets(r_cur,0,self.cube_side_length,mags_per_endring_zero_pos[ii],0,'')
                point_list = point_list + endring
                
            for jj in range(z_half):
                if(placement_ring_decision[ii,jj]==1):
                    z_pos = np.sum(increment_z_matrix[ii,:(jj+1)]) 
                    endring = magsimulator.generate_ring_of_magnets(r_cur,z_pos,self.cube_side_length,num_mags_matrix[ii,jj],phase_diff_mat[ii,jj],'')
                    point_list = point_list + endring
                
                    z_neg = -np.sum(increment_z_matrix[ii,:(jj+1)]) 
                    endring = magsimulator.generate_ring_of_magnets(r_cur,z_neg,self.cube_side_length,num_mags_matrix[ii,jj],phase_diff_mat[ii,jj],'')
                    point_list = point_list + endring        

        tmp_df = pd.DataFrame(point_list,columns =['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Searched','CostValue','Used','Placement_index','Bmag','Magnet_length','Tag'])

        cols_to_round = ['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Searched','CostValue','Used','Placement_index','Bmag','Magnet_length']
        tmp_df[cols_to_round] = tmp_df[cols_to_round].round(3)

               
        mag_vect = [1320,0,0] # N42 Br 13,200 G

        col_sensors = magpy.Collection(style_label='sensors')
        sensor1 = magsimulator.define_sensor_points_on_sphere(100,15,[0,0,0]) # 30 mm DSV
        col_sensors.add(sensor1)
                        
        magnets = magpy.Collection(style_label='magnets')
        # print(tmp_df.shape)
        if(tmp_df.shape[0]!=0):
            eta, meanB0,magnets,_ = magsimulator.simulate_ledger(magnets,col_sensors,mag_vect,tmp_df,0.06,4,True,False,None,False) 
            
        else:
            eta=1e6
            meanB0=0
        
        print('mean B0=' + str(np.round(meanB0,4)) + ' homogen=' + str(np.round(eta,4)))

        f1=eta
        f2=-meanB0
        # print(increment_z_matrix)
        # g = np.max(np.sum(increment_z_matrix[:,0:z_half],axis=1))-self.maxzextent/2
        
        # print(len(magnets))
        g2 = len(magnets)-self.maxnumberofmagnets
        out["F"] = np.column_stack([f1,f2])
        out["G"] = g2
        # out["G"] = np.column_stack([g,g2])
        
    def get_magnets(self,x):

        layers=np.array(x[f"x01"].reshape((1,1))).astype(int).flatten()
        layers=layers[0]
        zpositions=np.array(x[f"x02"]).reshape((1,1)).astype(int)
        zpositions=zpositions[0]
        mags_per_endring_zero_pos=np.array([x[f"x{k:02}"] for k in range(3, 8)]).reshape((5,1)).astype(int).flatten()
        increment_z_matrix=np.array([x[f"x{k:02}"] for k in range(8, 8+self.increment)]).reshape((self.increment,1)).flatten()
        num_mags_matrix=np.array([x[f"x{k:02}"] for k in range(8+self.increment, 8+self.increment*2)]).reshape((self.increment,1)).astype(int).flatten()
        placement_ring_decision=np.array([x[f"x{k:02}"] for k in range(8+self.increment*2, 8+self.increment*3)]).reshape((self.increment,1)).astype(int).flatten()
        phase_diff_mat=np.array([x[f"x{k:02}"] for k in range(8+self.increment*3, 8+self.increment*4)]).reshape((self.increment,1)).flatten()
        # print(layers.shape)
        # print(zpositions.shape)
        # print(mags_per_endring_zero_pos.shape)
        # print(increment_z_matrix.shape)
        # print(num_mags_matrix.shape)
        #constants
        
        increment_z_matrix=increment_z_matrix.reshape((self.maxlayers,self.maxzpos))
        num_mags_matrix=num_mags_matrix.reshape((self.maxlayers,self.maxzpos))
        placement_ring_decision = placement_ring_decision.reshape((self.maxlayers,self.maxzpos))
        phase_diff_mat = phase_diff_mat.reshape((self.maxlayers,self.maxzpos))
        
        if(np.mod(zpositions,2)==0):
            z_half = int(zpositions/2)
        else:
            z_half = int((zpositions-1)/2)
            
        point_list=[]
        # print(layers)
        # print(type(layers))
        for ii in range(layers):
            r_cur=self.rmin+ii*self.dr
            
            if(np.mod(zpositions,2)!=0):
                endring = magsimulator.generate_ring_of_magnets(r_cur,0,self.cube_side_length,mags_per_endring_zero_pos[ii],0,'')
                point_list = point_list + endring
                
            for jj in range(z_half):
                if(placement_ring_decision[ii,jj]==1):
                    z_pos = np.sum(increment_z_matrix[ii,:(jj+1)]) 
                    endring = magsimulator.generate_ring_of_magnets(r_cur,z_pos,self.cube_side_length,num_mags_matrix[ii,jj],phase_diff_mat[ii,jj],'')
                    point_list = point_list + endring
                
                    z_neg = -np.sum(increment_z_matrix[ii,:(jj+1)]) 
                    endring = magsimulator.generate_ring_of_magnets(r_cur,z_neg,self.cube_side_length,num_mags_matrix[ii,jj],phase_diff_mat[ii,jj],'')
                    point_list = point_list + endring        

        tmp_df = pd.DataFrame(point_list,columns =['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Searched','CostValue','Used','Placement_index','Bmag','Magnet_length','Tag'])

        cols_to_round = ['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Searched','CostValue','Used','Placement_index','Bmag','Magnet_length']
        tmp_df[cols_to_round] = tmp_df[cols_to_round].round(3)

               
        mag_vect = [1320,0,0] # N42 Br 13,200 G

        col_sensors = magpy.Collection(style_label='sensors')
        sensor1 = magsimulator.define_sensor_points_on_sphere(100,15,[0,0,0]) # 30 mm DSV
        col_sensors.add(sensor1)
                        
        magnets = magpy.Collection(style_label='magnets')
        # print(tmp_df.shape)
        eta, meanB0,_,_ = magsimulator.simulate_ledger(magnets,col_sensors,mag_vect,tmp_df,0.036,4,True,False,None,False) 
        print('mean B0=' + str(np.round(meanB0,4)) + ' homogen=' + str(np.round(eta,4)))

        return magnets, tmp_df
#%% 
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize

problem = MultiObjectiveMixedVariableProblem()

# algorithm = MixedVariableGA(pop_size=25, survival=RankAndCrowdingSurvival())

algorithm = NSGA2(pop_size=40,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  )

res = minimize(problem,
               algorithm,
               ('n_gen', N_GEN),
               seed=10,
               verbose=True)

filename = f'bore50mm_6p35mm_N42_maxmag{MAX_MAGNETS}_gen{N_GEN}'
rows=[]
for idx, xx in enumerate(np.atleast_1d(res.X)):
    mm, ledger = problem.get_magnets(xx)
    ledger.to_excel(f'results/{filename}_pareto{idx}.xlsx', index=False)
    rows.append({'pareto_idx': idx, 'magnets': len(ledger), 'layers': int(xx['x01']), 'homogeneity_ppm': res.F[idx,0], 'B0_mT': -res.F[idx,1],
                 'length_mm': ledger['Z-pos'].max()-ledger['Z-pos'].min()+problem.cube_side_length})
summary = pd.DataFrame(rows).sort_values('homogeneity_ppm')
summary.to_csv(f'results/{filename}_summary.csv', index=False)
print(summary.to_string())

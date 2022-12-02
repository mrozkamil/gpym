#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:00:15 2022

@author: km357
"""
from scipy.interpolate import splrep, splev
import numpy as np
import matplotlib.pyplot as plt
plt.ion()



class spline():
    def __init__(self, fine_nodes, coarse_nodes, deg = 3, make_plot = False):
        self.deg = deg
        self.internal_nodes = coarse_nodes
        self.fine_nodes = fine_nodes

        tmp_f = np.sin(np.linspace(0,2*np.pi, fine_nodes.size))

        self.tck = splrep(fine_nodes,tmp_f, task = -1,  t= coarse_nodes, k=deg)
        self.all_nodes = self.tck[0]

        self.all_node_cent = np.zeros(self.all_nodes.size)
        self.node_weigh = np.zeros(self.all_nodes.size)
        self.spline_base = np.zeros((fine_nodes.size, self.all_nodes.size, ))

        if make_plot:
            plt.figure()

        for ii in range(self.all_nodes.size):
            node_i = np.zeros_like(self.all_nodes)
            node_i[ii] =1.
            tck = (self.tck[0], node_i, self.tck[2])
            tmp_v = splev(fine_nodes,tck = tck, ext = 3)
            self.spline_base[:,ii] = tmp_v
            self.node_weigh[ii] = np.trapz(tmp_v,fine_nodes)
            self.all_node_cent[ii] = np.trapz(fine_nodes*tmp_v,fine_nodes)/self.node_weigh[ii]
            if make_plot:
                plt.plot(fine_nodes,tmp_v, color = 'C%d' % np.mod(ii,8))
                plt.plot(self.all_node_cent[ii],0., 'o', color = 'C%d' % np.mod(ii,8))
                plt.text(self.all_node_cent[ii],-0.1,
                         ' node %d \n W = %.2f' % (ii, self.node_weigh[ii]), 
                         color = 'C%d' % np.mod(ii,8))
        if make_plot:
            plt.grid()
            plt.show()
        
            
            
        self.node_signif = ~np.isclose(self.node_weigh,0.)
        self.ind_node_signif = np.where(self.node_signif)[0]
        self.node_cent = self.all_node_cent[self.ind_node_signif]

        print('input vector with %d nodes required' % (self.ind_node_signif.size))

    def __call__(self,node_values, der = 0):
        node_values_ext = np.zeros_like(self.all_nodes)
        node_values_ext[self.ind_node_signif] = node_values
        tck = (self.tck[0], node_values_ext, self.tck[2])
        return splev(self.fine_nodes,tck = tck, ext = 3, der = der)

    def get_rep(self, x,y):
        tmp =  splrep(x, y, task = -1,  t= self.internal_nodes, k=self.deg)[1]
        return tmp[self.node_signif]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:10:43 2023

@author: dsanchez
"""

import pickle
import towerpy as tp

fdir = '/home/enchiladaszen/Documents/mygithub/enchilaDaSzen/towerpy/datasets/'

with open(fdir+'vps.tpy', 'rb') as f:
    rprofs = pickle.load(f)

with open(fdir+'mlyrs.tpy', 'rb') as f:
    rmlyr = pickle.load(f)

radb = tp.datavis.rad_interactive.hti_base(rprofs, mlyrs=rmlyr,
                                           stats='std',
                                           var2plot='rhoHV [-]',
                                           htiylim=[0, 8], tz='Europe/London')
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)
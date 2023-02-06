#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:10:43 2023

@author: dsanchez
"""

import pickle
import towerpy as tp

fdir = '/home/enchiladaszen/Documents/mygithub/enchilaDaSzen/towerpy/datasets/'

with open(fdir+'qvps.tpy', 'rb') as f:
    rprofs = pickle.load(f)

with open(fdir+'mlyrsqvps.tpy', 'rb') as f:
    rmlyr = pickle.load(f)

radb = tp.datavis.rad_interactive.hti_base(rprofs, mlyrs=rmlyr,
                                           stats='std', ptype='fcontour',
                                           var2plot='rhoHV [-]',
                                           # contourl='ZH [dBZ]',
                                           htiylim=[0, 8], tz='Europe/London')
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)
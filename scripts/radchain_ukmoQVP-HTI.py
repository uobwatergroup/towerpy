"""Display an HTI plot from QVPs."""


import pickle
import towerpy as tp

rsite = 'chenies'
fdir = f'../datasets/{rsite}/y2020/spel4/'

with open(fdir+'qvps.tpy', 'rb') as f:
    rprofs = pickle.load(f)

with open(fdir+'mlyrsqvps.tpy', 'rb') as f:
    rmlyr = pickle.load(f)

for i in rprofs:
    i.profs_type = 'QVPs'

radb = tp.datavis.rad_interactive.hti_base(rprofs, mlyrs=rmlyr, stats='std',
                                           ptype='fcontour',
                                           var2plot='rhoHV [-]',
                                           # contourl='ZH [dBZ]',
                                           htiylim=[0, 8], tz='Europe/London')
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)

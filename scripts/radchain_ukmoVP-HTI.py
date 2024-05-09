"""Display an HTI plot from VPs."""

import pickle
import towerpy as tp

rsite = 'chenies'
fdir = f'../datasets/{rsite}/y2020/spel8/'

with open(fdir+'vps.tpy', 'rb') as f:
    rprofs = pickle.load(f)

with open(fdir+'mlyrs.tpy', 'rb') as f:
    rmlyr = pickle.load(f)

for i in rprofs:
    i.profs_type = 'VPs'

radb = tp.datavis.rad_interactive.hti_base(rprofs, mlyrs=rmlyr, stats='std',
                                           var2plot='rhoHV [-]',
                                           htiylim=[0, 8], tz='Europe/London')
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)

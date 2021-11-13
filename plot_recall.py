import os
import sys
import pickle
#import random

import matplotlib 
import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib import gridspec
#from scipy.special import softmax 
#from sklearn.metrics import confusion_matrix
#import pandas as pd
#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import reverse_cuthill_mckee
#from sklearn.preprocessing import normalize
matplotlib.use('Agg')
import seaborn as sn
plt.style.use('ggplot')
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
#matplotlib.rcParams.update({'font.size': 5})
#cat = 0
#if(len(sys.argv)>1):
#    cat = int(sys.argv[1])


#count=1
#with open("picklesaves/0E" +str(count) + ".pkl", "rb") as f:
#    E = pickle.load(f)
#with open("picklesaves/0tps_ids" +str(count) + ".pkl", "rb") as f:
#    tps_ids = pickle.load(f)
#with open("picklesaves/0fps_ids" +str(count) + ".pkl", "rb") as f:
#    fps_ids = pickle.load(f)
#
#print(tps_ids[:100])
#print(len(tps_ids))
#print(len(fps_ids))
#print(len(E))
#print(E[2])
#ids = [ x for e in E for x in e['dtIds'] ]
#print(len(ids))
#exit(0)

with open("precision1.pkl", "rb") as f:
    precision1 = pickle.load(f)

with open("precision5.pkl", "rb") as f:
    precision_ens = pickle.load(f)

with open("nmodel_precision5.pkl", "rb") as f:
    precision_nmodel = pickle.load(f)



cats = ['person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle']

single = [14.52, 15.25, 34.90, 17.58, 31.80, 9.64, 5.66, 7.62]
single4 = [19.52, 21.03, 43.06, 20.98, 38.71, 13.41, 11.27, 10.73]
ours4 = [19.52, 21.03, 43.06, 20.98, 38.71, 13.41, 11.27, 10.73]

width=0.2
x = np.arange(len(cats))

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.3, single, width, label='Single Capture')
rects2 = ax.bar(x - 0.1, single4, width, label='Single Capture (N Models)')
rects2 = ax.bar(x + 0.1, ours4, width, label='Multi Exposure Ensemble (Ours)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mAP')
#ax.set_title('mAP on CityScapes Dataset')
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.legend()

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

fig.tight_layout()
fig.savefig('saved_pdfs/map.pdf')

#plt.show()


#fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(12,5.8))
#for cat in range(8):
#for cat, ax in enumerate(axs.flat):
#    print(cat)
#    ax.set_aspect(0.75)
#    p1 = np.mean(precision1[0:1,:,cat,0,2], axis=0)
#    p_nmodel = np.mean(precision_nmodel[0:1,:,cat,0,2], axis=0)
#    p_ens = np.mean(precision_ens[0:1,:,cat,0,2], axis=0)
#    ax.plot(p1, np.linspace(0,1,num=101), label="Single Capture")
#    ax.plot(p_nmodel, np.linspace(0,1,num=101), label="Single Capture (N models)")
#    ax.plot(p_ens, np.linspace(0,1,num=101), label="Multi Exposure Ensemble (Ours)")
#    ax.set_title(cats[cat])
#    #ax.legend()
#    ax.set(xlabel='Recall', ylabel='Precision')
#    #ax.xlabel('Recall')
#    #ax.ylabel('Average Precision')
#    #ax.tight_layout()
#
#fig.tight_layout()
#fig.subplots_adjust(hspace=0., top=0.85)
#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels)
#
#for ax in axs.flat:
#    ax.label_outer()

#fig.savefig('saved_pdfs/IOU5.pdf')


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

n_bins = 4

labels = ['1', '2', '4', '8']
Unet_3_128 = [ 8* 646, 4* 646, 2* 646, 646]
Unet_4_128 = [ 8* 371, 4* 371, 2* 371, 371]
Unet_5_128 = [ 8* 354, 4* 354, 2* 354, 354]

fig, ax = plt.subplots()

#colors = ['red', 'tan', 'lime']

width = 0.75       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, Unet_3_128, width, label='Unet_3_128')
ax.bar(labels, Unet_4_128, width, label='Unet_4_128')
ax.bar(labels, Unet_5_128, width, label='Unet_5_128')
#ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
#               label='Women')

savefile_png = 'training_3D.png'
savefile_pdf = 'training_3D.pdf'


#plt.xscale("log")
plt.yscale("log")
plt.xlabel(r'N GPUs')
plt.ylabel(r'$t_{e}$  (s)')

#plt.yticks([250, 500, 1000, 2000])
#plt.xticks(cell_reduced, ('$16^3$', '$32^3$', '$64^3$', '$128^3$', '$192^3$'))
plt.ylim(100, 10000)
fig.tight_layout()

plt.savefig(savefile_png)
plt.savefig(savefile_pdf)



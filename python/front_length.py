"""
Calculate front length scale at wall.
Usage:
    front_length.py <files>... [--output=<dir>]

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from docopt import docopt
from dedalus.tools import logging
from dedalus.tools import post
from dedalus.tools.parallel import Sync

plt.style.use('prl')

def calc_front(filename, start, count, output=None):

    filename = pathlib.Path(filename)
    stem = filename.stem.split("_")[0]
    dpi = 150
    savename_func = lambda write: '{}_{:06}.png'.format(stem, write)
    # Layout

    fig = plt.figure(figsize=(10,5.5))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    cax = fig.add_axes([0.1,0.9,0.2,0.02])
    axes.set_xlabel(r'$k_x$')
    axes.set_ylabel(r'$z$')

    with h5py.File(filename,mode='r') as file:
        img = axes.imshow(np.zeros_like(file['tasks/Temperature(y=+1)'][0,:,:,0]))
        cbar = fig.colorbar(img, cax=cax,orientation='horizontal')

        for index in range(start, start+count):
            temp = file['tasks/Temperature(y=+1)'][index,:,:,0]
            c_temp = np.fft.rfft(temp,axis=1)
            power = (c_temp * c_temp.conj()).real
            log_power = np.log10(power)

            vmin = np.nanmin(log_power)
            vmax = np.nanmax(log_power)
            if vmin == -np.inf:
                vmin = vmax-16
            print("vmin, vmax = {},{}".format(vmin,vmax))
            img.set_data(log_power)
            cbar.mappable.set_clim(vmin=vmin,vmax=vmax)
            cbar.set_ticks([vmin,vmax])
            cax.xaxis.tick_top()
            cax.xaxis.set_label_position('top')

            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
    plt.close(fig)

args = docopt(__doc__)
output_path = pathlib.Path(args['--output']).absolute()
# Create output directory if needed
with Sync() as sync:
    if sync.comm.rank == 0:
        if not output_path.exists():
            output_path.mkdir()
post.visit_writes(args['<files>'], calc_front, output=output_path)
    

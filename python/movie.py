"""
Plot movie frames 
Usage:
    movie.py <files>... [--output=<dir> --slice-type=<slice-type>]

Options:
    --output=<dir>              Output directory [default: ./movie_snapshots]
"""
import re
import pathlib
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from dedalus.extras import plot_tools


def main(filename, start, count, output="./img_snapshots", slice_type="side"):
    """Save plot of specified tasks for given range of analysis writes."""

    filename = pathlib.Path(filename)
    stem = filename.stem.split("_")[0]
    tasks = 'Temperature(y=+1)'
    dpi = 150
    savename_func = lambda write: '{}_{:06}.png'.format(stem, write)
    # Layout

    fig = plt.figure(figsize=(10,5))
    axes = fig.add_axes([0, 0, 1, 1])
    axes.spines['right'].set_color('none')
    axes.spines['left'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.spines['bottom'].set_color('none')
    # turn off ticks
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    axes.xaxis.set_ticklabels([])
    axes.yaxis.set_ticklabels([])
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            axes.imshow(file['tasks/'+tasks][index,:,:,0], cmap='coolwarm')
                
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            axes.clear()
    plt.close(fig)


if __name__ == "__main__":

    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    slice_type = args['--slice-type']
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    print("slice_type = {}".format(slice_type))
    post.visit_writes(args['<files>'], main, output=output_path, slice_type=slice_type)


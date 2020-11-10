"""
Plot planes from joint analysis files.

Usage:
    plot_snapshots.py <files>... [--output=<dir> --slice-type=<slice-type>]

Options:
    --output=<dir>              Output directory [default: ./img_snapshots]
    --slice-type=<slice-type>   plot type [default: side]
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

    if (slice_type == "top"):
        nrows, ncols = 2,2
        image_axes = [2, 3]
        data_slices = [0, 0,slice(None), slice(None)]
        tasks = ['Pressure(z=+1)', 'Ux(z=+1)', 'Uy(z=+1)']
    elif (slice_type == "mid"):
        nrows, ncols = 2,2
        image_axes = [2, 3]
        data_slices = [0, 0,slice(None), slice(None)]
        tasks = ['W(z=+0.5)', 'T(z=+0.5)']
    elif (slice_type == "side"):
        nrows, ncols = 2,2
        image_axes = [2, 1]
        data_slices = [0, slice(None), slice(None),0]
        tasks = ['Temperature(y=+1)', 'Pressure(y=+1)', 'Ox(y=+1)', 'Oz(y=+1)']
    elif (slice_type == 'small'):
        nrows, ncols = 2,2
        image_axes = [2, 1]
        data_slices = [0, slice(None), slice(None),0]
        tasks = ['Temperature(y=+1)', 'Pressure(y=+1)', 'Oz(y=+1)']
    else:
        raise ValueError("must be side, mid or top slice. got {}".format(str(filename)))
    
    # Plot settings

    scale = 4
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: '{}_{:06}.png'.format(stem, write)
    # Layout

    image = plot_tools.Box(3, 1)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                data_slices[0] = index
                plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
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


"""
Dedalus script for 3D wall-mode convection.

[L] = Lz
[T] = Lz^2/kappa

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 wall_modes.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

"""
import sys
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

from configparser import ConfigParser
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

config_file = Path(sys.argv[-1])
config = ConfigParser()
config.read(str(config_file))

logger.info('Running with the following parameters:')
logger.info(config.items('parameters'))

params = config['parameters']

# Parameters
Nz = params.getint('Nz')
Nx = params.getint('Nx')
Ny = params.getint('Ny')
Procz = params.getint('Procz')
Procx = params.getint('Procx')

criticality    = params.getfloat('criticality')
Pr             = params.getfloat('Pr')
Ek             = params.getfloat('Ek')

box_wavenumber = params.getfloat('box_wavenumber')
time_step_freq = params.getfloat('time_step_freq')

stop_sim_time  = params.getfloat('stop_sim_time')
if stop_sim_time is None:
    stop_sim_time = np.inf

stop_wall_time = params.getfloat('stop_wall_time')
if stop_wall_time is None:
    stop_wall_time = 25*60*60

stop_iteration = params.getint('stop_iteration')
if stop_iteration is None:
    stop_iteration = np.inf

logger.info("stop_sim_time: {}; stop_wall_time: {}; stop_iteration: {}".format(stop_sim_time, stop_wall_time, stop_iteration))
Amp            = 1e-3

pi             =  np.pi
Rc             =  np.sqrt(6*np.sqrt(3))*(pi**2)
Lc             =  np.sqrt(6) - np.sqrt(2)
kc             =  2*pi/Lc
Racb           =  3*(np.pi**2/2)**(2/3)*Ek**(-4/3.)


R              =  criticality*Rc
Ra             =  R/Ek
Lx, Ly, Lz     =  box_wavenumber*Lc, 1., 1.

alpha          =  (kc*R)/(kc**2+pi**2)
beta           =  (kc/pi)*alpha
omega          =  2*alpha*beta                       # oscilation frequency
gamma          =  beta**2 - kc**2 - pi**2 - alpha**2 # growth rate
tau            =  2*pi/omega

dt0            =  time_step_freq*tau


logger.info('wall mode growth rate: %f, frequency: %f' %(gamma,omega))
logger.info('wall mode supercriticality: %f' % criticality)
logger.info('bulk supercriticality: %f' % (Ra/Racb))

# Create bases and domain
z_basis =    de.SinCos('z',Nz, interval=   (0, Lz), dealias=3/2)
x_basis =   de.Fourier('x',Nx, interval=   (0, Lx), dealias=3/2)
y_basis = de.Chebyshev('y',Ny, interval= (-Ly, Ly), dealias=3/2)
domain = de.Domain([z_basis,x_basis,y_basis], grid_dtype=np.float64, mesh=[Procz,Procx])

# 3D Wall Modes
problem = de.IVP(domain, variables=['p','u','v','w','T','ox','oz','Q'])

# turbo mode for chebyshev solve.
problem.meta['p','u','v','w','Q']['y']['dirichlet'] = True

# +1 for cosine basis. -1 for sine basis.
problem.meta['u','oz','v','p']['z']['parity'] =  1
problem.meta['w','ox','T','Q']['z']['parity'] = -1

problem.parameters['R']  = R
problem.parameters['E0'] = Ek
problem.parameters['E1'] = Ek/Pr
problem.parameters['Area']   = 2*Lx*Ly
problem.parameters['Volume'] = 2*Lx*Ly*Lz

problem.substitutions['r(thing)'] = 'right(thing)'
problem.substitutions['l(thing)'] = 'left(thing)'

problem.substitutions['oy'] = ' dz(u) - dx(w)'

problem.add_equation("E1*dt(u) -   v + dx(p) + E0*( dy(oz) - dz(oy) ) = E1*( oz*v - oy*w )")
problem.add_equation("E1*dt(v) +   u + dy(p) + E0*( dz(ox) - dx(oz) ) = E1*( ox*w - oz*u )")
problem.add_equation("E1*dt(w) - R*T + dz(p) + E0*( dx(oy) - dy(ox) ) = E1*( oy*u - ox*v )")
problem.add_equation(" dx(u) + dy(v) + dz(w) = 0")

problem.add_equation(" oz - dx(v) + dy(u) = 0 ")
problem.add_equation(" ox - dy(w) + dz(v) = 0 ")

problem.add_equation(" dt(T) -  w - ( d(T,x=2) + dy(Q) + d(T,z=2) ) = - (u*dx(T) + v*Q + w*dz(T) )")
problem.add_equation(" Q - dy(T) = 0 ")

problem.add_bc("l(v) = 0", condition="(nx!=0) or  (nz!=0)")
problem.add_bc("l(p) = 0", condition="(nx==0) and (nz==0)")
problem.add_bc("r(v) = 0")
problem.add_bc("l(u) = 0")
problem.add_bc("r(u) = 0")
problem.add_bc("l(w) = 0")
problem.add_bc("r(w) = 0")
problem.add_bc("l(Q) = 0")
problem.add_bc("r(Q) = 0")

# Build solver
#solver = problem.build_solver(de.timesteppers.MCNAB2)
solver = problem.build_solver(de.timesteppers.SBDF4)
logger.info('Solver built')

# Initial conditions
T, Q = solver.state['T'], solver.state['Q']
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
T['g'] =  Amp*noise
# easy filter.
T.set_scales(1/4, keep_data=True)
T['c']
T['g']
T.set_scales(1, keep_data=True)
T.differentiate('y', out=Q)

# Integration parameters
solver.stop_sim_time  = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Analysis
datadir = Path("scratch") / config_file.stem
#
if domain.dist.comm.rank == 0:
    if not datadir.exists():
        datadir.mkdir()

slices = solver.evaluator.add_file_handler(datadir/Path('slices'), sim_dt=dt0, max_writes=50)
# side wall
slices.add_task("interp(T,y=+1)",name='Temperature(y=+1)')
slices.add_task("interp(p,y=+1)",name='Pressure(y=+1)')
slices.add_task("interp(ox,y=+1)",name='Ox(y=+1)')
slices.add_task("interp(oz,y=+1)",name='Oz(y=+1)')

# top
slices.add_task("interp(p,z=+1)", name='Pressure(z=+1)')
slices.add_task("interp(u,z=+1)", name='Ux(z=+1)')
slices.add_task("interp(v,z=+1)", name='Uy(z=+1)')

# mid
slices.add_task("interp(w,z=+0.5)",name='W(z=+0.5)')
slices.add_task("interp(T,z=+0.5)",name='T(z=+0.5)')

data = solver.evaluator.add_file_handler(datadir/Path('data'), iter=100, max_writes=np.inf)
data.add_task("integ(w*T)/Volume",name='Nusselt')
data.add_task("0.5*integ(u*u)/Volume",name='Energy_x')
data.add_task("0.5*integ(v*v)/Volume",name='Energy_y')
data.add_task("0.5*integ(w*w)/Volume",name='Energy_z')

data.add_task("0.5*integ(ox*ox)/Volume",name='Enstrophy_x')
data.add_task("0.5*integ(oy*oy)/Volume",name='Enstrophy_y')
data.add_task("0.5*integ(oz*oz)/Volume",name='Enstrophy_z')

checkpoints = solver.evaluator.add_file_handler(datadir/Path('checkpoints'), wall_dt=3540, max_writes=1)
checkpoints.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=0.01*dt0, cadence=10, safety=0.3,
                     max_change=1.5, min_change=0.5, max_dt=dt0)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=50)
flow.add_property("0.5*(u*u)", name='Energy_x')
flow.add_property("0.5*(v*v)", name='Energy_y')
flow.add_property("0.5*(w*w)", name='Energy_z')
flow.add_property("0.5*(ox*ox)", name='Enstrophy_x')
flow.add_property("0.5*(oy*oy)", name='Enstrophy_y')
flow.add_property("0.5*(oz*oz)", name='Enstrophy_z')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 50 == 0:
            Kx, Ky, Kz = flow.max('Energy_x'),    flow.max('Energy_y'),    flow.max('Energy_z')
            Ex, Ey, Ez = flow.max('Enstrophy_x'), flow.max('Enstrophy_y'), flow.max('Enstrophy_z')
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max K: %e, %e, %e' %(Kx, Ky, Kz))
            logger.info('Max E: %e, %e, %e' %(Ex, Ey, Ez))
            if np.isnan(Ex):
                logger.info('NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN')
                break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    logger.info('DOF cycles/cpu sec: %f' % (Nz*Ny*Nx*solver.iteration/(end_time-start_time)/domain.dist.comm_cart.size))

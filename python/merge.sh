echo "running with $1 cores"
conda activate dedalus-dev
mpirun -np $1 python3 -m dedalus merge_procs top_slices
mpirun -np $1 python3 -m dedalus merge_procs side_slices
mpirun -np $1 python3 -m dedalus merge_procs mid_slices

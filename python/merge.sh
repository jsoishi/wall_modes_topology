echo "running with $1 cores"
conda activate dedalus-dev
mpirun -np $1 python3 merge.py $2/slices --cleanup
mpirun -np $1 python3 merge.py $2/checkpoints --cleanup
mpirun -np $1 python3 merge.py $2/data --cleanup

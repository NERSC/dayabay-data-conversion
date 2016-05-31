# Scripts to Convert Raw Daya Bay Root Files to HDF5 Files.

## Usage on Cori:

### MPI4PY version:
* sbatch -N 12 mpi_submit_extract_all.sl 

### Job array version:
* sbatch submit_extract_all.sl

Testing:
* source setup_make_dataset.sh
* cd extract_ibd
* python extract_ibd_from_yasu.py

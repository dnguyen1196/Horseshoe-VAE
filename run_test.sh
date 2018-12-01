#!/usr/bin/bash
#module load python/3.5.0
source activate bdl_pytorch_readonly


# This is the option to specify for the server
# -c number of cores -mem memory --time (right now is 4 days)
# How this gets called is

# cat input_file |  thisfile.sh
# The input_file should contain a bunch of hyperparameters in each line

# That means the python script must also be called like this
# python -u test_vae.py --vae $vae_type --nns $nns --ncode $ncode
# So it understands these command line arguments. This can be done using
# ArgumentParser

opts="-p batch  -c 2 --mem=4196 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # outdirectory
read vae_type # Type of VAE
read nns    # Neural network structure
read ncode  # Number of latent code
read data   # data set

nnsfile="${nns// /_}"
python_interpreter="/cluster/tufts/hugheslab/miniconda2/envs/bdl_pytorch_readonly/bin/python"

echo "$outdir$vae_type-$nnsfile-$ncode.out"

# Specify the name of the output file
# These output files will be used to for plotting the metrics
outs="--output=$outdir$vae_type-$nnsfile-$ncode.out --error=$outdir$vae_type-$nnsfile-$ncode.err"

sbatch $opts $outs --wrap="$python_interpreter -u test_vae.py --vae $vae_type --nns $nns --ncode $ncode --data $data"


source deactivate


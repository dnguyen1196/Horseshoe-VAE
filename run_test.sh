#!/usr/bin/bash
module load python/3.5.0

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


while IFS=' ' read -r data model
do
    # Specify the name of the output file
    # These output files will be used to for plotting the metrics
    outs="--output=$outdir$vae_type-$nns-$ncode.out --error=$outdir$vae_type-$nns-$ncode.err"
    sbatch $opts $outs --wrap="./venv/bin/python -u test_vae.py --vae $vae_type --nns $nns --ncode $ncode"
    sleep 1
done
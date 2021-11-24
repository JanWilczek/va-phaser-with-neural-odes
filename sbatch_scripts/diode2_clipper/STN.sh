#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=3700M
#SBATCH --output=logs/diode2_clipper/stn-%j.txt

# hidden size 40 needs around 3GB and 50 hours for 1200 epochs + test
# hidden size 20 needs just under 7 hours to complete for 400 epochs + test
HIDDEN_SIZE=20

module load gcc
module load cuda
module load miniconda
source activate vavnode

set -x

srun python diode2_clipper/main.py --method STN --layers_description "3x${HIDDEN_SIZE}x2" --epochs 400 --batch_size 25 --up_fr 100 --val_chunk 22050 --test_chunk 0 --learn_rate 0.0005 --teacher_forcing always --dataset_name diode2clip --validate_every 10 --loss_function ESRLoss --nonlinearity Tanh --name Hidden$HIDDEN_SIZE

# srun --time=00:05:00 --mem=1000M --gres=gpu:1 --constraint=volta python diode2_clipper/main.py --method STN --layers_description 3x20x20x20x2 --epochs 100 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.000001 --teacher_forcing always --dataset_name diode2clip --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Tanh --name Hidden20

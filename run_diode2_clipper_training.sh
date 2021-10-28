#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1500M

# srun python diode2_clipper/main.py --method LSTM --batch_size 64 --init_len 1000 --epochs 600 --learn_rate 0.001 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --dataset_name diode2clip --hidden_size 16 --loss_function ESR_DC_prefilter --state_size 2 --validate_every 10 --checkpoint October27_16-00-11_dgx5.int.triton.aalto.fi

# srun python diode2_clipper/main.py --method ScaledODENetFE --epochs 600 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.01 --exponential_lr 0.0001 --teacher_forcing always --dataset_name diode2clip --hidden_size 30 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLPFE

# srun python diode2_clipper/main.py --method ScaledODENetFE --epochs 600 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.01 --exponential_lr 0.0001 --teacher_forcing always --dataset_name diode2clip --hidden_size 20 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLP2FE

# srun python diode2_clipper/main.py --method forward_euler --epochs 600 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.01 --exponential_lr 0.0001 --teacher_forcing always --dataset_name diode2clip --hidden_size 30 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLP

# srun python diode2_clipper/main.py --method forward_euler --epochs 600 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.01 --exponential_lr 0.0001 --teacher_forcing always --dataset_name diode2clip --hidden_size 20 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLP2

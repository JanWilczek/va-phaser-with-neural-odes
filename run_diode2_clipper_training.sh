HIDDEN_SIZE=30
set -x

for TEST_SAMPLING_RATE in 44100
do
python diode2_clipper/main.py --method ScaledODENetRK4 --epochs 1 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size $HIDDEN_SIZE --validate_every 1 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLPRK4 --name "Hidden${HIDDEN_SIZE}" --test_sampling_rate $TEST_SAMPLING_RATE

python diode2_clipper/main.py --method ScaledODENetMidpoint --epochs 100 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size 30 --validate_every 5 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLPRK4 --name "Hidden30" --test_sampling_rate 44100
done

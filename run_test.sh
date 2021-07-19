CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py --method forward_euler --batch_size 512 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 1 --learn_rate 0.001 --teacher_forcing bernoulli --name TEST_RUN --test_sampling_rate 22050 --save_sets
rm -rv diode_clipper/runs/diodeclip/forward_euler/*TEST_RUN
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 512 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 1 --learn_rate 0.001 --teacher_forcing bernoulli --name TEST_RUN --test_sampling_rate 22050
rm -rv phaser/runs/forward_euler/*TEST_RUN

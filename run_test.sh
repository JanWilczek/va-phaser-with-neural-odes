EPOCHS=1
METHOD=odeint_euler
DEVICE=0
CUDA_VISIBLE_DEVICES=$DEVICE python diode_clipper/main.py --method $METHOD --batch_size 512 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001 --teacher_forcing bernoulli --name TEST_RUN --test_sampling_rate 22050 --dataset_name diodeclip
rm -rv diode_clipper/runs/diodeclip/`echo "$METHOD" | tr '[:upper:]' '[:lower:]'`/*TEST_RUN
CUDA_VISIBLE_DEVICES=$DEVICE python phaser/main.py --method $METHOD --batch_size 512 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001 --teacher_forcing bernoulli --name TEST_RUN --test_sampling_rate 22050 --dataset_name FameSweetToneOffNoFb
rm -rv phaser/runs/FameSweetToneOffNoFb/`echo "$METHOD" | tr '[:upper:]' '[:lower:]'`/*TEST_RUN

set -x
output=all_models.txt
> $output
for folder in diode_clipper diode2_clipper
do
  for file in 44100.txt 22050.txt 48000.txt 192000.txt
  do
    cat $folder/$file >> $output
  done
done

#python scripts/compute_nmse_sdr.py `cat ${output}`

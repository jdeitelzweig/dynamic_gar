# dynamic_gar
Spring 2021 IW 

To fine tune a model, run
```
python fine_tune.py
  --model_name_or_path <model_name> \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file <train_filepath> \
  --validation_file <validation_filepath> \
  --test_file <test_filepath> \
  --output_dir <path_to_output_dir> \
  --overwrite_output_dir \
  --text_column "question" \
  --summary_column "augment" \
  --predict_with_generate \ 
  --learning_rate 0.00005 \
  --num_train_epochs 15 \
  --sc_scaling 0.98 \
```

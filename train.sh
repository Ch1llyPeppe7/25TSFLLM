#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 17 --lm_layer_num 6 --lm_ft_type freeze --clip 5  OOM
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 17 --lm_layer_num 5 --lm_ft_type freeze --clip 5 OOM
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 17 --lm_layer_num 4 --lm_ft_type freeze --clip 5 OOM
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 17 --lm_layer_num 3 --lm_ft_type freeze --clip 5 NaN
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 17 --lm_layer_num 3 --lm_ft_type freeze --clip 3 NaN
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 17 --lm_layer_num 3 --lm_ft_type freeze --clip 5 --learning_rate 1e-5 NaN
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 17 --lm_layer_num 3 --lm_ft_type fpt --clip 5 --learning_rate 1e-5 NaN
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 14 --lm_layer_num 2 --lm_ft_type freeze --clip 3 --learning_rate 1e-5 --mask_rate 0.25 --patch_len 32 ok 
#python run.py --gpu 0 --training_list execute_list/train.csv --max_token_num 14 --lm_layer_num 2 --lm_ft_type freeze --clip 3 --learning_rate 1e-5 --mask_rate 0.25 --patch_len 32 worse
#python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 14 --lm_layer_num 2 --lm_ft_type fpt --clip 5 --learning_rate 1e-5 --mask_rate 0.25 --patch_len 32 fpt
python run.py --gpu 0 --training_list execute_list/train_all.csv  --max_token_num 15 --lm_layer_num 2 --lm_ft_type freeze --clip 5 --learning_rate 1e-5 --mask_rate 0.5 --patch_len 32 


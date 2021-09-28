# declare -A cri_dict=(['A']='month' ['B']='week')
# declare -A maxW_dict=(['week']=20 ['month']=8)
# declare -A name_dict=(['madgan']='madgan')

# target_name='A'
# criterion=${cri_dict[${target_name}]}
# max_window=${maxW_dict[$criterion]}

# # model params
# gpu_i=0
# n_epochs=150
# learning_rate=0.01
# p_num=2
# arch='madgan'

for index in `seq 10000 10000 200000`
do
    python main.py \
            --mode test \
            --test_mode sep \
            --visible_gpus 0 \
            --window_size 3 \
            --test_from models/index_A03/model_w3_fixed_step_$index.pt \
            --data_type bbc_news \
            --test_sep_num -1 \
            --test_max_mode max_one \
            --threshold 0.8 \
            --add_transformer
done

# for index in `seq 0 2 10`
# do
#     echo print_$index.pt
# done
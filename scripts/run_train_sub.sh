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
for ws in 1 2 3 4 5
do
    echo window size: $ws
    python main.py \
            --mode train \
            --random_seed 227182 \
            --window_size $ws \
            --batch_size 3000 \
            --save_checkpoint_steps 10000 \
            --train_steps 150000 \
            --valid_steps 20000 \
            --visible_gpus 1 \
            --backbone_type bert \
            --add_transformer True \
            --finetune_bert True \
            --classifier_type linear \
            --lr 0.0001 \
            --warmup_steps 10000 \
            --model_index NE0$ws
done
# for index in `seq 0 2 10`
# do
#     echo print_$index.pt
# done
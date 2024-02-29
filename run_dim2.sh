#!/bin/bash
ssl_type=wavlm-large
current_datetime=$(date +'%Y%m%d-%H%M')

# Train
pool_type=AttentiveStatisticsPooling
for seed in 7; do
    python train_ft_dim_ser.py \
        --seed=${seed} \
        --ssl_type=${ssl_type} \
        --batch_size=32 \
        --accumulation_steps=4 \
        --lr=1e-5 \
        --epochs=20 \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/wavLM_adamW/${current_datetime}/${seed} || exit 0;
    
    # Evaluation on Test3 and save results using with format required by challenge
python eval_dim_ser_test3.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/wavLM_adamW/${current_datetime}/${seed}  \
        --store_path=result/dim_ser/wavLM_adamW/${current_datetime}/${seed}.txt || exit 0;

    # general evaluation
    python eval_dim_ser.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/wavLM_adamW/${current_datetime}/${seed}  \
        --store_path=result/dim_ser/wavLM_adamW/${current_datetime}${seed}.txt || exit 0;
    
done

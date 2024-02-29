#!/bin/bash
ssl_type=wavlm-large

# Train
# avalilable pooling types: AttentiveStatisticsPooling, MeanPooling, TemporalStatisticsPooling
# MinPooling, MaxPooling, MinMaxPooling, SelfAttentivePooling
# pool_type=AttentiveStatisticsPooling
pool_type=MeanPooling
batch=32
for seed in 7; do
    python train_ft_dim_ser.py \
        --seed=${seed} \
        --ssl_type=${ssl_type} \
        --batch_size=${batch} \
        --accumulation_steps=4 \
        --lr=1e-5 \
        --epochs=30 \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/wavLM_adamW/${pool_type}/${batch}/${seed} || exit 0;
    
    # Evaluation on Test3 and save results using with format required by challenge
    python eval_dim_ser_test3.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/wavLM_adamW/${pool_type}/${batch}/${seed}  \
        --store_path=result/dim_ser/wavLM_adamW/${pool_type}/${batch}/${seed}.txt || exit 0;

    # general evaluation
    python eval_dim_ser.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/wavLM_adamW/${pool_type}/${batch}/${seed} \
        --store_path=result/dim_ser/wavLM_adamW/${seed}.txt || exit 0;
    
done

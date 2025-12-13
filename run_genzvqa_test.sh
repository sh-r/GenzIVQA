accelerate config
accelerate launch genzvqa_test.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
        --train_batch_size 16 --val_batch_size 1 \
        --output_dir checkpoints  --num_train_epochs 6 \
        --learning_rate 1e-4 \
        --sampling_time_steps 4 --train_data LSVQ --val_data NA --test_data KoNViD \
        --accelerator_ckpts_dir checkpoints/genzvqa_exp1/accelerator_states \
        --unet_ckpts_dir checkpoints/genzvqa_exp1/unet_states \
        --test_res_save_path results/genzvqa_exp1 \
        --coop_ckpts_dir checkpoints/genzvqa_exp1/coop_states \
        --fastAttn_ckpts_dir checkpoints/genzvqa_exp1/fastAttention \
        --slowAttn_ckpts_dir checkpoints/genzvqa_exp1/slowAttention \
        --linBlock_ckpts_dir checkpoints/genzvqa_exp1/linearBlocks \
        --calculate_srocc --best_validation_epoch 2
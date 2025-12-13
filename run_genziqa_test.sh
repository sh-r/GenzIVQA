accelerate config
accelerate launch genziqa_test.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
        --train_batch_size 16 --val_batch_size 1 \
        --output_dir checkpoints  --num_train_epochs 10 \
        --learning_rate 1e-4 \
        --sampling_time_steps 8 --train_data FLIVE --val_data NA --test_data CLIVE \
        --accelerator_ckpts_dir checkpoints/genziqa_exp1/accelerator_states \
        --unet_ckpts_dir checkpoints/genziqa_exp1/unet_states \
        --test_res_save_path results/genziqa_exp1 \
        --coop_ckpts_dir checkpoints/genziqa_exp1/coop_states \
        --best_validation_epoch 4
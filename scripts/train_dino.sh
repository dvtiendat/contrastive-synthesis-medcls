python -m scripts.run_dino \
    --init_with_imagenet True \
    --config configs/contrastive/dino.yaml \
    --data_path sample_data \
    --output_dir sample_data \
    --epochs 100 \
    --batch_size 1 \
    --learning_rate 0.0005
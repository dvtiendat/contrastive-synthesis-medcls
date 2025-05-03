python -m scripts.run_classification \
    --config configs/classification/finetune_dino.yaml \
    --data_path /path/to/your/labeled/dataset \
    --output_dir /path/to/save/finetune/results \
    --pretrained_checkpoint_path /path/to/dino/checkpoint/best_model.pth \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --epochs 50 \
    --freeze_backbone False
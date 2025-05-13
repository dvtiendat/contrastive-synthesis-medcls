# python -m scripts.run_classification_vit \
#     --config configs/classification/finetune_dino.yaml \
#     --data_path /path/to/your/labeled/dataset \
#     --output_dir /path/to/save/finetune/results \
#     --pretrained_checkpoint_path /path/to/dino/checkpoint/best_model.pth \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --epochs 50 \
#     --freeze_backbone False

python -m scripts.run_classification_vit \
    --config configs/classification/finetune_dino_rd.yaml \
    --data_path sample_data \
    --output_dir results \
    --pretrained_checkpoint_path ckpts/best_model.pth \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --epochs 1 \
    --freeze_backbone False \
    --pretrain_source random
python -m scripts.evaluate_classification \
    --config configs/classification/finetune_dino.yaml \
    --checkpoint_path /path/to/your/best_finetune_model.pth \
    --data_path /path/to/your/data \
    --output_dir results/test_evaluation 
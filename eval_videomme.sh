accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model internvl2 \
    --model_args pretrained="OpenGVLab/InternVL2-8B",modality="video" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "InternVL2-8B" \
    --output_path ./logs/
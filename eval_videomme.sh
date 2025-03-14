accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    --model internvl2 \
    --model_args pretrained="OpenGVLab/InternVL2_5-8B",modality="video",enable_shared_cross_attention=False \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "InternVL25-8B" \
    --output_path ./logs/


for local_attention_group_size in 8; do
    accelerate launch --num_processes 1 --main_process_port 12346 -m lmms_eval \
        --model internvl2 \
        --model_args "pretrained=OpenGVLab/InternVL2_5-8B,modality=video,enable_shared_cross_attention=True,local_attention_group_size=${local_attention_group_size}" \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "InternVL25-8B-local-attn-$local_attention_group_size" \
        --output_path ./logs/
done


for num_frame in 32 64 128; do
    accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
        --model internvl2 \
        --model_args pretrained="OpenGVLab/InternVL2_5-8B",modality="video",enable_shared_cross_attention=False,num_frames=$num_frame \
        --tasks longvideobench_val_v \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "InternVL25-8B" \
        --output_path ./logs/
done

for num_frame in 64; do
    for local_attention_group_size in 2 4 8 16 32; do
        accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
            --model internvl2 \
            --model_args "pretrained=OpenGVLab/InternVL2_5-8B,modality=video,enable_shared_cross_attention=True,local_attention_group_size=258*${local_attention_group_size},num_frames=$num_frame" \
            --tasks longvideobench_val_v \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "InternVL25-8B-local-attn-$local_attention_group_size" \
            --output_path ./logs/
    done
done
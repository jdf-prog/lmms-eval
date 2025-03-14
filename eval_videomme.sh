accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    --model internvl2 \
    --model_args pretrained="OpenGVLab/InternVL2_5-8B",modality="video",enable_shared_cross_attention=False \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "InternVL25-8B" \
    --output_path ./logs/


num_processes=4
for num_frame in 32; do
    for local_attention_group_size in 4 8 32; do
        for top_k in 100 -1; do
            echo "num_frame: $num_frame, local_attention_group_size: $local_attention_group_size, top_k: $top_k"
            accelerate launch --num_processes ${num_processes} --main_process_port 12346 -m lmms_eval \
                --model internvl2 \
                --model_args "pretrained=OpenGVLab/InternVL2_5-8B,modality=video,enable_shared_cross_attention=True,local_attention_group_size=${local_attention_group_size},num_frames=$num_frame,top_k=$top_k" \
                --tasks videomme \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix "InternVL25-8B-local-attn-$local_attention_group_size-topk-$top_k-frames-$num_frame" \
                --output_path ./logs/
        done
    done

    # accelerate launch --num_processes ${num_processes} --main_process_port 12345 -m lmms_eval \
    #     --model internvl2 \
    #     --model_args pretrained="OpenGVLab/InternVL2_5-8B",modality="video",enable_shared_cross_attention=False,num_frames=$num_frame \
    #     --tasks videomme \
    #     --batch_size 1 \
    #     --log_samples \
    #     --log_samples_suffix "InternVL25-8B" \
    #     --output_path ./logs/
done


for num_frame in 32; do
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
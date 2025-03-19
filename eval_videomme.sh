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


num_processes=1
for num_frame in 32; do
    for local_attention_group_size in 16; do
        for top_k in 100; do
            echo "num_frame: $num_frame, local_attention_group_size: $local_attention_group_size, top_k: $top_k"
            accelerate launch --num_processes ${num_processes} --main_process_port 12348 -m lmms_eval \
                --model internvl2 \
                --model_args "pretrained=OpenGVLab/InternVL2_5-8B,modality=video,enable_shared_cross_attention=True,adaptive_local_attention=True,local_attention_group_size=${local_attention_group_size},num_frames=$num_frame,top_k=$top_k" \
                --tasks videomme \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix "InternVL25-8B-local-attn-$local_attention_group_size-topk-$top_k-frames-$num_frame" \
                --output_path ./logs/
        done
    done
done


num_processes=1
for local_attention_group_size in -1; do
    for top_k in 16 32 64 100 -1; do
        for predict_type in key_norms_small key_norms vector_norms vector_norms_small; do 
            echo "local_attention_group_size: $local_attention_group_size, top_k: $top_k, predict_type: $predict_type"
            accelerate launch --num_processes ${num_processes} --main_process_port 12346 -m lmms_eval \
                --model internvl2 \
                --model_args "pretrained=OpenGVLab/InternVL2_5-8B,modality=image,enable_shared_cross_attention=True,local_attention_group_size=${local_attention_group_size},top_k=$top_k,predict_type=$predict_type" \
                --tasks mmmu_val \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix "InternVL25-8B-local-attn-$local_attention_group_size-topk-$top_k-frames-$num_frame-predict_type-$predict_type" \
                --output_path ./logs/
        done
    done
done

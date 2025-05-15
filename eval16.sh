export FORCE_QWENVL_VIDEO_READER='decord'
adaptive_local_attention=True
num_processes=8
benchmark_name=videomme
for num_frame in 256; do
    for local_attention_group_size in 16; do
        for top_p in 0.1 0.2 0.4 0.6 0.8 0.9; do
            for predict_type in key_norms_small; do 
                for top_k_starting_layer in 0; do
                    for prune_during_prefill_layer_idx in -1; do
                        echo "num_frame: $num_frame, local_attention_group_size: $local_attention_group_size, top_p: $top_p, predict_type: $predict_type, top_k_starting_layer: $top_k_starting_layer, prune_during_prefill_layer_idx: $prune_during_prefill_layer_idx"
                        accelerate launch --num_processes ${num_processes} --main_process_port 12351 -m lmms_eval \
                            --model qwen2_5_vl \
                            --model_args "pretrained="Qwen/Qwen2.5-VL-7B-Instruct",max_num_frames=$num_frame,use_flash_attention_2=True,adaptive_local_attention=$adaptive_local_attention,local_attention_group_size=${local_attention_group_size},top_p=$top_p,predict_type=$predict_type,top_k_starting_layer=$top_k_starting_layer,prune_during_prefill_layer_idx=$prune_during_prefill_layer_idx" \
                            --tasks $benchmark_name \
                            --batch_size 1 \
                            --log_samples \
                            --log_samples_suffix "Qwen2.5-VL-7B-Instruct-frames-$num_frame-local_attention_group_size-$local_attention_group_size-top_p-$top_p-predict_type-$predict_type-top_k_starting_layer-$top_k_starting_layer-prune_during_prefill_layer_idx-$prune_during_prefill_layer_idx" \
                            --output_path ./logs/qwen2_5_vl_${num_frame}_var_top_p
                    done
                done
            done
        done
    done
done
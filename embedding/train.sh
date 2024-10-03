CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir /output \
--model_name_or_path /model/bge \
--train_data /data/train.jsonl \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 32 \
--passage_max_len 256 \
--train_group_size 10 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval ""


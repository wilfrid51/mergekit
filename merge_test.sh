export RAY_record_task_actor_creation_sites=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"
export HF_ALLOW_CODE_EVAL="1"
export confirm_run_unsafe_code=True
mergekit-evolve \
  --storage-path ./evolve_storage \
  examples/evolve_config.yml \
  --vllm \
  --wandb \
  --wandb-project merge_4b \
  --wandb-entity yamalies-ajio \
  --task-search-path tasks \
  --i-understand-the-depths-of-the-evils-i-am-unleashing
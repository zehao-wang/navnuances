
mkdir -p ./cache # caching images of current standing position

python NavGPT4v.py --llm_model_name gpt-4-turbo --root_dir ./data \
    --output_dir ./data/R2R/exprs/gpt-4v-turbo
MEGATRON_PATH= # The path of Megatron-LM 
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# If you meet the error "layer.mlp.shared_experts.gate_weight.data.copy_(hf_layer.mlp.shared_expert_gate.weight)
#                               AttributeError: 'NoneType' object has no attribute 'data'"
# Make sure you have change the "False" to "True" at Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py#240
# shared_experts=ModuleSpec(
#     module=SharedExpertMLP,
#     params={"gate": False}, -> params={"gate": True},
#     submodules=MLPSubmodules(
#         linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
#         linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
#     ),
# )

# This converter only support convert qwen2 HF ckpt to Mcore legacy sequential mlp ckpt
# Which means, if you want to load the converted ckpt, you should not add "--moe-grouped-mlp"
# To generate dist-ckpt from the legacy ckpt, please use '--auto-detect-ckpt-format ' to load the ckpt with megatron, then save it.
HF_FORMAT_DIR= # The path of hugging face ckpt
TOKENIZER_MODEL= # The path of Qwen2 Tokenizer
MEGATRON_FORMAT_DIR= # The path of converted ckpt
python Qwen2Converter/checkpoint/convert.py \
--model-type GPT \
--loader qwen2_hf \
--saver mcore \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL} \
--target-tensor-parallel-size 4 \
--target-pipeline-parallel-size 4 \
--target-expert-parallel-size 1 \
--megatron-path ${MEGATRON_PATH} \
--saver-transformer-impl transformer_engine 

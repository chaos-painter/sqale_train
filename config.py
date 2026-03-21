# Model Settings
MODEL_NAME = "unsloth/Qwen3-8B-bnb-4bit"
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

# LoRA Settings
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Training Hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
MAX_STEPS = 60
OUTPUT_DIR = "outputs"
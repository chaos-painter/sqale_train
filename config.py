# Model Settings
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

# LoRA Settings
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Training Hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
MAX_STEPS = 150
OUTPUT_DIR = "outputs"


# Prompt Settings
SQL_PROMPT = """Below is a database schema and a natural language question. Write a SQL query to answer the question accurately.

### Schema:
{}

### Question:
{}

### Response:
{}"""
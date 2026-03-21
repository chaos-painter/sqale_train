from unsloth import FastLanguageModel
import config

def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_NAME,
        max_seq_length = config.MAX_SEQ_LENGTH,
        load_in_4bit = config.LOAD_IN_4BIT,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = config.LORA_ALPHA,
        lora_dropout = config.LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )
    return model, tokenizer
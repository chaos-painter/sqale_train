from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import config
from model_utils import load_model_and_tokenizer
from data_utils import get_sqale_dataset

def main():
    model, tokenizer = load_model_and_tokenizer()
    dataset = get_sqale_dataset(tokenizer)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = config.MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = config.BATCH_SIZE,
            gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate = config.LEARNING_RATE,
            max_steps = config.MAX_STEPS,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = config.OUTPUT_DIR,
            optim = "adamw_8bit",
        ),
    )

    trainer.train()
    model.save_pretrained("qwen3_sqale_final")
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()
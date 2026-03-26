from unsloth import FastLanguageModel
import json
import os
from config import SQL_PROMPT

# 1. Load your Unsloth adapter and base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-0.8B", # Tell it EXACTLY what to download
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Load your LORA adapter on top of it
model = FastLanguageModel.for_inference(model) # Enable inference mode
model.load_adapter("adapter") # Point to your folder
FastLanguageModel.for_inference(model) # 2x faster inference!

# 2. Load the Spider dev questions
with open("spider_data/spider_data/dev.json", "r") as f:
    dev_data = json.load(f)

predictions = []

# 3. Loop through and generate
for item in dev_data:
    db_id = item["db_id"]
    question = item["question"]
    
    # Path to schema.sql
    schema_path = f"spider_data/spider_data/database/{db_id}/schema.sql"
    
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_info = f.read().strip()
    else:
        schema_info = "Schema not found."

    # 4. FIXED: Format the prompt correctly using the template variables
    # We leave the "Response" part empty for the model to fill in.
    prompt = SQL_PROMPT.format(schema_info, question, "")
    
    # Tokenize and Generate
    inputs = tokenizer(text = [prompt], return_tensors = "pt").to("cuda")
    
    # We use 'max_new_tokens' to ensure the model doesn't stop too early
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the NEW tokens (the model's answer)
    predicted_sql = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Spider eval needs single lines
    clean_sql = predicted_sql.replace('\n', ' ').strip()
    predictions.append(clean_sql)

# 5. Save the predictions to a text file for the official Spider evaluator
with open("predicted_spider.txt", "w", encoding="utf-8") as f:
    for sql in predictions:
        f.write(sql + "\n")

print("Done! Predictions saved to predicted_spider.txt")
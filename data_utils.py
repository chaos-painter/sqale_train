from datasets import load_dataset

SQL_PROMPT = """Below is a database schema and a natural language question. Write a SQL query to answer the question accurately.

### Schema:
{}

### Question:
{}

### Response:
{}"""

def get_sqale_dataset(tokenizer):
    def formatting_prompts_func(examples):
        texts = [
            SQL_PROMPT.format(s, q, r) + tokenizer.eos_token
            for s, q, r in zip(examples["schema"], examples["question"], examples["query"])
        ]
        return {"text": texts}

    dataset = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")
    return dataset.map(formatting_prompts_func, batched=True)
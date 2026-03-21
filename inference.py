from unsloth import FastLanguageModel
from data_utils import SQL_PROMPT
import torch

def run_test(schema, question):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qwen3_sqale_final",
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer([SQL_PROMPT.format(schema, question, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.batch_decode(outputs))

if __name__ == "__main__":
    run_test("CREATE TABLE students (id INT, grade INT)", "Show me students with grade 10")
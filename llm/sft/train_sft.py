"""Minimal SFT training skeleton using LoRA PEFT.
Fill dataset loading and model selection before running.
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def main():
    model_name = "gpt2"
    ds = load_dataset("imdb", split="train[:1%]")  # placeholder dataset
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, peft_cfg)

    def collate(batch):
        texts = [x["text"] for x in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc["labels"] = enc["input_ids"].clone()
        return enc

    args = TrainingArguments(
        output_dir="outputs/sft",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=3e-4,
        logging_steps=5,
        save_steps=50,
        fp16=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.save_pretrained("outputs/sft/model")

if __name__ == "__main__":
    main()

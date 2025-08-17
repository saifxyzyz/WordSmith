from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset
from main import model, tokenizer
import torch
from context_compressor import ContextCompressor
dataset1 = 'timdettmers/openassistant-guanaco'
dataset2 = 'allenai/sciq' 


if __name__ == '__main__':
    def compress_text(example):
        compressor = ContextCompressor()
        # We assume the text is in the 'text' key.
        # This function compresses the text and returns a dictionary with the compressed text.
        # It's crucial to pass the text argument to the compress method correctly.
        compressed_text_obj = compressor.compress(example['text'], target_ratio=0.5)
        return {'text': compressed_text_obj.compressed_text}
    training_args = TrainingArguments(
    output_dir="./gemma3_model_output",
    per_device_train_batch_size=2, # Keep this small for memory
    fp16=True, # if you have a powerful GPU remove this line and insert bf16 = True
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=100,
    save_strategy="epoch",
    )

      
    


    original_dataset = load_dataset('json', data_files = 'custom_dataset.json', split='train')
    compressed_dataset = original_dataset.map(compress_text, num_proc=2)
# # Now, initialize SFTTrainer without the 'tokenizer' keyword argument
    trainer = SFTTrainer(
        model=model,
        train_dataset=original_dataset,
        args=training_args, # Unpack the SFTConfig object
    )
    trainer.train()

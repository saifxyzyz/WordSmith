import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

device = "cpu"  #change to cuda if have GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  #comment this line if you have GPU

# Define the paths for the base model and your fine-tuned adapter
# Use the same model ID from Hugging Face for the base model
BASE_MODEL = "unsloth/gemma-3-270m-it"
# This is the directory where SFTTrainer saved your model
ADAPTER_PATH = "gemma3_model_output\checkpoint-36" 

# --- Load the base model ---
# For CPU, we load the model in full precision (float32) for best performance
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=device,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("Base model loaded successfully.")

# --- Load the fine-tuned adapter weights ---
print("Loading PEFT adapters...")
# Use PeftModel.from_pretrained to load your LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH, local_files_only=True)

# Merging the adapter weights into the base model is optional but recommended for inference
# It makes the model faster by combining the weights into a single, cohesive model
# and eliminates the need for the PEFT library at inference time.
model = model.merge_and_unload()
print("Adapters merged successfully.")

#compile the model
model = torch.compile(model)

# --- Define a prompt for the model to generate text from ---
# Use the same chat template that the model was fine-tuned on
prompt = str(input("Ask WordSmith: ")) 
messages = [
    {"role": "user", "content": prompt}
]
chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Generate the response ---
print("\nThinking...")
inputs = tokenizer(chat_template, return_tensors="pt").to(device)

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=500,  # Adjust for desired output length
    do_sample=True,
    temperature=0.5,
    top_k=50,
    top_p=0.75,
)

# Decode and print the output
decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
# The output will include the prompt and the model's response.
# We extract the model's response by finding the end of the input prompt in the output.
response = decoded_output.split("model\n")[-1].strip()
print("\n--- Model Response ---")
print(response)

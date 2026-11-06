import torch
from transformers import pipeline

# 1. Target your Nvidia GPU for execution
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Initializing LLM execution layer on: {'GPU (Quadro T2000)' if device == 0 else 'CPU'}")

# 2. Select a lightweight, high-performance local instruct model
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

print(f"⏳ Loading '{model_id}' into VRAM...")
generator = pipeline("text-generation", model=model_id, device=device)

# 3. Format the prompt structure using standard ChatML chat templates
messages = [
    {"role": "system", "content": "You are a helpful and concise AI assistant."},
    {"role": "user", "content": "Explain what a GPU is in exactly one short sentence."}
]

# Apply the template wrapper to the text string
prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n🤖 Querying local LLM agent...")
# 4. Generate text with explicit hyperparameters controlling randomness
outputs = generator(
    prompt, 
    max_new_tokens=50, 
    temperature=0.7, 
    top_p=0.9,
    do_sample=True
)

# Extract and clean up the text response payload
generated_text = outputs[0]["generated_text"]
response_only = generated_text[len(prompt):].strip()

print("\n💬 LLM Response:")
print("-" * 60)
print(response_only)
print("-" * 60)

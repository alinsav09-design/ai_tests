import torch
from transformers import pipeline

# 1. Set execution hardware target
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Initializing Chat Bot on: {'GPU (Quadro T2000)' if device == 0 else 'CPU'}")

# 2. Load the interactive local model
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
print(f"⏳ Loading '{model_id}' into VRAM...")
generator = pipeline("text-generation", model=model_id, device=device)

# 3. Initialize conversation history with system rules
messages = [
    {"role": "system", "content": "You are a helpful, friendly, and concise local AI assistant."}
]

print("\n🤖 Local Chatbot Initialized! Type 'exit' or 'quit' to end the conversation.")
print("-" * 70)

# 4. Interactive conversation loop
while True:
    user_input = input("\nYou: ")
    
    # Check for breakout commands
    if user_input.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_input.strip():
        continue
        
    # Append the user prompt to the active session history
    messages.append({"role": "user", "content": user_input})
    
    # Render chat template format string
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("AI is thinking...")
    outputs = generator(
        prompt, 
        max_new_tokens=150, 
        temperature=0.7, 
        top_p=0.9,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    # Extract only the newly generated text block
    generated_text = outputs[0]["generated_text"]
    response_only = generated_text[len(prompt):].strip()
    
    # Print the model output and save it back to history for context retention
    print(f"\nAI: {response_only}")
    messages.append({"role": "assistant", "content": response_only})

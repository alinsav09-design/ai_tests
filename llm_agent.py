import torch
from transformers import pipeline
from datetime import datetime
import requests

# 1. Target Hardware Setup
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Initializing Smart AI Agent on: {'GPU (Quadro T2000)' if device == 0 else 'CPU'}")

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
generator = pipeline("text-generation", model=model_id, device=device)

# --- Python Tools Definitions ---

def get_current_date():
    """Tool: Returns the exact current date and time."""
    now = datetime.now()
    return f"The current date and time is: {now.strftime('%A, %B %d, %Y at %I:%M %p')}."

def get_weather_romania():
    """Tool: Fetches the live current temperature for Romania (Hunedoara area) via open-meteo API."""
    try:
        # Lat/Lon for Hunedoara, Romania region
        url = "https://open-meteo.com"
        response = requests.get(url, timeout=5).json()
        current = response.get("current_weather", {})
        temp = current.get("temperature", "unknown")
        wind = current.get("windspeed", "unknown")
        return f"The current live weather in Hunedoara, Romania is {temp}°C with a wind speed of {wind} km/h."
    except Exception:
        return "The live weather data server is currently unreachable, but it is definitely not June right now!"

# --- Conversation Loop ---

messages = [
    {"role": "system", "content": "You are a factual, concise local AI assistant. Use the provided tools context to answer accurately."}
]

print("\n🤖 Local Agent with Live Tools Initialized! Type 'exit' to quit.")
print("-" * 75)

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_input.strip():
        continue

    # --- Intent Detection Router (The Agent Layer) ---
    tool_context = ""
    lowered_input = user_input.lower()
    
    if "date" in lowered_input or "month" in lowered_input or "time" in lowered_input or "year" in lowered_input:
        print("🔧 [Agent System] Triggering Tool: get_current_date()...")
        tool_context = get_current_date()
        
    elif "weather" in lowered_input or "temperature" in lowered_input:
        print("🔧 [Agent System] Triggering Tool: get_weather_romania()...")
        tool_context = get_weather_romania()

    # If a tool was triggered, augment the prompt with ground truth data before giving it to the LLM    
    if tool_context:
        # --- ENHANCED STRICT ISOLATION WRAPPER ---
        agent_augmented_prompt = (
            f"### CRITICAL GROUND TRUTH DATA:\n"
            f"```text\n{tool_context}\n```\n"
            f"INSTRUCTION: You must strictly use the data above to answer the user's question. "
            f"Ignore all previous months or seasons mentioned in chat history.\n"
            f"User Question: {user_input}"
        )
        messages.append({"role": "user", "content": agent_augmented_prompt})
    else:
        messages.append({"role": "user", "content": user_input})

        
    # Render final ChatML prompt layout
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("AI is compiling answer...")
    outputs = generator(
        prompt, 
        max_new_tokens=100, 
        temperature=0.3, # Lower temperature means more deterministic adherence to our tool context
        top_p=0.9,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    generated_text = outputs[0]["generated_text"]
    response_only = generated_text[len(prompt):].strip()
    
    print(f"\nAI: {response_only}")
    
    # Save clean history turn
    messages.append({"role": "assistant", "content": response_only})

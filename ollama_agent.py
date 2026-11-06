import ollama
from datetime import datetime
import requests

MODEL_ID = "llama3.2:1b"

print(f"🚀 Initializing Advanced Ollama Agent Layer using '{MODEL_ID}'...")

# --- Python Tools Definitions ---

def get_current_date():
    """Tool: Returns the exact current date and time."""
    now = datetime.now()
    return f"The current date and time is: {now.strftime('%A, %B %d, %Y at %I:%M %p')}."

def get_weather_romania():
    """Tool: Fetches the live current temperature for Hunedoara, Romania via open-meteo API."""
    try:
        # Lat/Lon for Hunedoara, Romania
        url = "https://open-meteo.com"
        response = requests.get(url, timeout=5).json()
        current = response.get("current_weather", {})
        temp = current.get("temperature", "unknown")
        wind = current.get("windspeed", "unknown")
        return f"The current live weather in Hunedoara, Romania is {temp}°C with a wind speed of {wind} km/h."
    except Exception:
        return "The live weather data server is currently unreachable. Tell the user you cannot fetch real-time weather due to a local network timeout."

# --- Conversation History ---

messages = [
    {
        "role": "system",
        "content": (
            "You are a factual, precise local AI assistant. When the user asks about the date, time, "
            "or weather, you will be provided with a context block labeled '### GROUND TRUTH DATA'. "
            "You must strictly use that data to answer and ignore any conflicting assumptions."
        )
    }
]

print("\n🤖 Smart Local Agent Active! Type 'exit' to quit.")
print("-" * 75)

# --- Conversational Loop ---
while True:
    user_input = input("\nYou: ")
    
    if user_input.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_input.strip():
        continue

    # --- Intent Routing (The Agent Layer) ---
    tool_context = ""
    lowered_input = user_input.lower()
    
    if any(word in lowered_input for word in ["date", "month", "time", "year", "day"]):
        print("🔧 [Agent System] Triggering Tool: get_current_date()...")
        tool_context = get_current_date()
        
    elif any(word in lowered_input for word in ["weather", "temperature", "degrees"]):
        print("🔧 [Agent System] Triggering Tool: get_weather_romania()...")
        tool_context = get_weather_romania()

    # If an external tool was triggered, isolate it into a strict instruction context
    if tool_context:
        agent_prompt = (
            f"### GROUND TRUTH DATA:\n"
            f"```text\n{tool_context}\n```\n"
            f"User Question: {user_input}"
        )
        messages.append({"role": "user", "content": agent_prompt})
    else:
        messages.append({"role": "user", "content": user_input})
        
    print("AI is compiling answer...")
    
    try:
        # Stream the smarter model's output in real time
        response_stream = ollama.chat(
            model=MODEL_ID,
            messages=messages,
            stream=True,
            options={"temperature": 0.2} # Low temperature ensures strict factual alignment with the tools
        )
        
        print("\nAI: ", end="", flush=True)
        full_response = ""
        
        for chunk in response_stream:
            text_chunk = chunk['message']['content']
            print(text_chunk, end="", flush=True)
            full_response += text_chunk
            
        print()
        
        # Save structural turn history
        messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        print(f"\n❌ Error execution failure: {e}")

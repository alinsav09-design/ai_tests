import ollama

# 1. Target the model you just downloaded via the Ollama CLI
MODEL_ID = "llama3.2:1b"

print(f"🚀 Connecting to local Ollama server running '{MODEL_ID}'...")

# 2. Initialize the system prompt and chat history array
messages = [
    {
        "role": "system",
        "content": "You are a highly capable, concise, and helpful local AI assistant running directly on the user's hardware."
    }
]

print("\n🤖 Local Ollama Chatbot Active! Type 'exit' or 'quit' to end the session.")
print("-" * 75)

# 3. Conversational Loop
while True:
    user_input = input("\nYou: ")
    
    if user_input.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_input.strip():
        continue
        
    # Append the user's input to the chat history state
    messages.append({"role": "user", "content": user_input})
    
    print("AI is generating response...")
    
    try:
        # 4. Stream the response chunks from the model for a real-time typing effect
        response_stream = ollama.chat(
            model=MODEL_ID,
            messages=messages,
            stream=True
        )
        
        print("\nAI: ", end="", flush=True)
        full_response = ""
        
        # Iterate over the live text streams as they are computed on your GPU
        for chunk in response_stream:
            text_chunk = chunk['message']['content']
            print(text_chunk, end="", flush=True)
            full_response += text_chunk
            
        print() # New line after the stream completes
        
        # Save the full assistant response to history to maintain long-term context
        messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        print(f"\n❌ Error connecting to Ollama server: {e}")
        print("Make sure the Ollama background service is running.")

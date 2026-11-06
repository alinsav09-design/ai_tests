import torch
from transformers import pipeline

# 1. Force GPU execution
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Modelul va rula pe: {'GPU (Quadro T2000)' if device == 0 else 'CPU'}")

# 2. Specify a Multilingual Sentiment model that explicitly understands Romanian
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

print(f"⏳ Downloading and loading '{model_name}' into VRAM...")
analizor = pipeline("sentiment-analysis", model=model_name, device=device)

# 3. Our test sentences
texte = [
    "Îmi place la nebunie acest bootcamp de AI, totul funcționează perfect pe placa mea video!",
    "Eroarea de sistem de mai devreme a fost extrem de enervantă și frustrantă.",
    "Mergem mai departe cu următorul capitol din cod."
]

print("\n📊 Actual Romanian Sentiment Results:")
print("-" * 50)

# 4. Run the model over the texts
for text in texte:
    rezultat = analizor(text)[0]
    label = rezultat['label']  # This model outputs labels like '1 star', '5 stars'
    scor = rezultat['score'] * 100
    
    # Map the star labels to clear sentiment meanings
    stars = int(label.split()[0])
    if stars >= 4:
        sentiment = "✅ POSITIVE (Good/Excellent)"
    elif stars == 3:
        sentiment = "😐 NEUTRAL"
    else:
        sentiment = "❌ NEGATIVE (Bad/Frustrating)"
    
    print(f"Text: \"{text}\"")
    print(f"Predicție: {sentiment} ({label} - Confidence: {scor:.2f}%)")
    print("-" * 50)

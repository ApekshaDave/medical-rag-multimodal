from google import genai

# 1. Initialize the new Client
# Make sure your key is EXACTLY as it appears in AI Studio
client = genai.Client(api_key="[ENCRYPTION_KEY]")

print("Checking available models with the NEW SDK...")

try:
    # In the new SDK, we use client.models.list()
    for m in client.models.list():
        # Check if it supports the generate_content method
        if 'generate_content' in m.supported_actions:
            print(f"✅ Found: {m.name}")
except Exception as e:
    print(f"❌ Still getting an error: {e}")
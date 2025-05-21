from transformers import pipeline

# prompt = "Today I learned,"
prompt = "Weirdly, I discovered"

# model = "EleutherAI/gpt-neo-125M"
model = "EleutherAI/gpt-neo-1.3B"

generator = pipeline("text-generation", model=model)
for _ in range(5):
    response = generator(prompt, do_sample=True, min_length=21, max_length=48)
    print(f"{response[0]['generated_text']}")

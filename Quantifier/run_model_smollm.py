from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "HuggingFaceTB/smollm2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "User: Explain python in programming.\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False   # 🔥 disables randomness
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
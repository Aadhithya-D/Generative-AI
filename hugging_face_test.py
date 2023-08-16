import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")
model = AutoModelForCausalLM.from_pretrained(
    "upstage/Llama-2-70b-instruct-v2",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

prompt = "### User:\nThomas is healthy, but he has to go to the hospital. What could be the reasons?\n\n### Assistant:\n"


# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
#
# tokenizer = AutoTokenizer.from_pretrained("succinctly/text2image-prompt-generator")
# model = AutoModelForCausalLM.from_pretrained("succinctly/text2image-prompt-generator")
# prompt = "Black shirt and white pant"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
del inputs["token_type_ids"]
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
#
# prompt2 = f"https://aiblocksnft.herokuapp.com/midjourneydiffusion/?prompt={output_text}&width=512&height=512&num_outputs=1&num_inference_steps=50&guidance_scale=7.5&seed=-1"
# print(output_text)
# print(prompt2)


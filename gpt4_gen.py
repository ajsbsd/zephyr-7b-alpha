from flask import Flask, request, jsonify
import torch
from transformers import pipeline

# Initialize the Zephyr-7B-alpha model
pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-alpha",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Create Flask app
app = Flask(__name__)

@app.route("/", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 200)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        result = pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
        return jsonify({"response": result[0]["generated_text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


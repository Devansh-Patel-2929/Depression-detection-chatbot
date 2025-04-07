# app.py
from flask import Flask, render_template, request, session
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from llama_cpp import Llama
import re
from huggingface_hub import hf_hub_download


import sys
import os
os.environ['FLASK_NO_COLOR'] = '1'
os.environ['TERM'] = 'dumb'
sys.stderr = sys.__stderr__  # Bypass custom error streams
sys.stdout = sys.__stdout__




app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

# Initialize models
def load_models():
    # Load depression model
    depression_model_path = "./depression_bert_model"
    depression_tokenizer = AutoTokenizer.from_pretrained(depression_model_path)
    depression_model = BertForSequenceClassification.from_pretrained(depression_model_path)
    depression_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load LLM
    llm_model_path = hf_hub_download(
        repo_id="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        filename="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        revision="main"
    )
    llm = Llama(
        model_path=llm_model_path,
        n_ctx=4096,
        n_gpu_layers=35,
        verbose=False
    )
    
    return depression_tokenizer, depression_model, llm

depression_tokenizer, depression_model, llm_model = load_models()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()

def predict_depression(text):
    cleaned = clean_text(text)
    inputs = depression_tokenizer(
        cleaned,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding='max_length'
    )
    inputs = {k: v.to(depression_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = depression_model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()

def format_response(raw_text):
    # Convert markdown-like formatting to HTML
    formatted = raw_text.replace('**', '').replace('__', '')

    # Add emojis for a more engaging response
    formatted = formatted.replace("sadness", "😢 sadness")
    formatted = formatted.replace("friends", "👫 friends")
    formatted = formatted.replace("family", "👨‍👩‍👧‍👦 family")
    formatted = formatted.replace("professional help", "🧑‍⚕️ professional help")
    formatted = formatted.replace("joy", "😊 joy")
    formatted = formatted.replace("exercise", "🏃‍♂️ exercise")
    formatted = formatted.replace("creative pursuits", "🎨 creative pursuits")
    formatted = formatted.replace("volunteering", "🤝 volunteering")

    # Add line breaks for better spacing
    formatted = '\n'.join([line.strip() for line in formatted.split('\n') if line.strip()])

    # Add bullet points for suggestions
    if "suggestions" in raw_text.lower():
        formatted = formatted.replace("* ", "• ")

    return formatted


def generate_response(user_input, history):
    # Define allowed topics
    system_messages = {
        'high': "Provide empathetic support with 1-2 concise paragraphs.",
        'moderate': "Respond with 1 brief paragraph.",
        'moderate': "keep conversation open for response.",
        'low': "Keep responses under 2 sentences."
    }

    depression_prob = predict_depression(user_input)
    risk_level = 'high' if depression_prob > 0.7 else 'moderate' if depression_prob > 0.4 else 'low'

    chatml_prompt = f"<|im_start|>system\n{system_messages[risk_level]}<|im_end|>\n"
    for msg in history[-4:]:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        chatml_prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"

    chatml_prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    response = llm_model(
        prompt=chatml_prompt,
        max_tokens=150,  # Shortened response length
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|im_end|>"]
    )

    raw_response = response['choices'][0]['text'].split('<|im_end|>')[0].strip()
    return format_response(raw_response)


@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'history' not in session:
        session['history'] = []
    
    if request.method == 'POST':
        user_input = request.form['message']
        if user_input.strip():
            session['history'].append({'role': 'user', 'content': user_input})
            
            response = generate_response(user_input, session['history'])
            session['history'].append({'role': 'assistant', 'content': response})
            
            session.modified = True
    
    return render_template('chat.html', chat_history=session['history'])

@app.route('/reset', methods=['POST'])
def reset_chat():
    session['history'] = []
    return render_template('chat.html', chat_history=session['history'])

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=False, use_debugger=False)
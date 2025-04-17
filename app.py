# app.py
from flask import Flask, render_template, request, session, redirect, url_for
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from llama_cpp import Llama
import re
from huggingface_hub import hf_hub_download
import sys
import os

# Environment configuration for Flask output
os.environ['FLASK_NO_COLOR'] = '1'
os.environ['TERM'] = 'dumb'
sys.stderr = sys.__stderr__
sys.stdout = sys.__stdout__

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

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
        n_gpu_layers=35 if torch.cuda.is_available() else 0,
        verbose=False
    )
    
    return depression_tokenizer, depression_model, llm

depression_tokenizer, depression_model, llm_model = load_models()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()

def limit_history(history, max_length=20):
    return history[-max_length:] if len(history) > max_length else history

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
    formatted = raw_text.replace('**', '').replace('__', '')
    replacements = {
        "sadness": "😢 sadness",
        "friends": "👫 friends",
        "family": "👨👩👧👦 family",
        "professional help": "🧑⚕️ professional help",
        "joy": "😊 joy",
        "exercise": "🏃♂️ exercise",
        "creative pursuits": "🎨 creative pursuits",
        "volunteering": "🤝 volunteering"
    }
    for word, replacement in replacements.items():
        formatted = formatted.replace(word, replacement)
    return '\n'.join([line.strip() for line in formatted.split('\n') if line.strip()])

def generate_response(user_input, history):
    system_messages = {
        'high': "Provide empathetic support with 1-2 concise paragraphs.",
        'moderate': "Respond with 1 brief paragraph and keep conversation open.",
        'low': "Keep responses under 2 sentences."
    }

    depression_prob = predict_depression(user_input)
    risk_level = 'high' if depression_prob > 0.7 else 'moderate' if depression_prob > 0.4 else 'low'

    chatml_prompt = f"<|im_start|>system\n{system_messages[risk_level]}<|im_end|>\n"
    for msg in limit_history(history, 4):
        role = 'user' if msg['role'] == 'user' else 'assistant'
        chatml_prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"

    chatml_prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    response = llm_model(
        prompt=chatml_prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|im_end|>"]
    )

    raw_response = response['choices'][0]['text'].split('<|im_end|>')[0].strip()
    return format_response(raw_response)

@app.route('/', methods=['GET', 'POST'])
def main_chat():
    if 'history' not in session:
        session['history'] = []
        
    if request.method == 'POST':
        user_input = request.form.get('message', '').strip()
        if user_input:
            session['history'].append({'role': 'user', 'content': user_input})
            response = generate_response(user_input, session['history'])
            session['history'].append({'role': 'assistant', 'content': response})
            session['history'] = limit_history(session['history'])
            session.modified = True
    return render_template('chat.html', chat_history=session['history'])

@app.route('/reset', methods=['POST'])
def reset_chat():
    session.clear()
    return redirect(url_for('main_chat'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

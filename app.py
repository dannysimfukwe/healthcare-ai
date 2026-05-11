from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import httpx
import json

app = FastAPI(title="Healthcare AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    model: Optional[str] = "phi3:latest"
    messages: Optional[List[Dict]] = []
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    language: str
    confidence: float

AI_API_URL = os.getenv("AI_API_URL", "https://ai-api.42helv.com/v1")
AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "https://ai-api.42helv.com")
AI_TYPE = os.getenv("AI_TYPE", "medical_assistant")
BASE_URL = os.getenv("SITE_BASE_URL", "")
API_KEY = os.getenv("AI_API_KEY", "")

AI_TITLES = {
    "medical_assistant": "Medical Assistant",
    "clinic_copilot": "Clinic Copilot",
    "transcription": "Transcription System",
    "triage": "Triage System",
    "pharmacy": "Pharmacy Assistant",
    "hospital_search": "Hospital Search"
}

LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - AI Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {{ font-family: 'Inter', sans-serif; }}
        .gradient-bg {{ background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }}
        .message-user {{ background: #f97316; color: white; }}
        .message-ai {{ background: #fff7ed; color: #9a3412; }}
        .text-orange-100 {{ color: rgba(255, 237, 213, 0.8); }}
        .text-orange-700 {{ color: #c2410c; }}
        .text-orange-900 {{ color: #9a3412; }}
        .bg-orange-50 {{ background-color: #fff7ed; }}
        .bg-orange-600 {{ background-color: #ea580c; }}
        .border-orange-200 {{ border-color: #fed7aa; }}
        .bg-orange-600:hover {{ background-color: #c2410c; }}
        button.gradient-bg {{ background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }}
        input:focus {{ border-color: #f97316; }}
        .typing-dot {{ animation: typing 1.4s infinite; }}
        @keyframes typing {{ 0%, 60%, 100% {{ opacity: 0.3; }} 30% {{ opacity: 1; }} }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <header class="bg-black text-white py-6">
        <div class="max-w-6xl mx-auto px-4 flex justify-between items-center">
            <div>
                <h1 class="text-2xl font-bold">{title}</h1>
                <p class="text-orange-400 text-sm mt-1">AI-Powered Assistant</p>
            </div>
            <a href="{repo_url}" target="_blank"
               class="flex items-center gap-2 bg-orange-500 hover:bg-orange-600 px-4 py-2 rounded-lg transition">
                <i class="fab fa-github"></i>
                <span>Download Code</span>
            </a>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-8">
        <div class="grid md:grid-cols-2 gap-8">
            <div>
                <div class="bg-white rounded-xl shadow-lg overflow-hidden">
                    <div class="gradient-bg px-4 py-3 text-white">
                        <div class="flex items-center gap-3">
                            <div class="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                                <i class="fas fa-robot"></i>
                            </div>
                            <div>
                                <h2 class="font-semibold">{title}</h2>
                                <p class="text-xs text-orange-100">Try it now</p>
                            </div>
                        </div>
                    </div>
                    <div class="chat-container p-4 flex flex-col h-96">
                        <div class="flex gap-2 mb-3">
                            <input type="password" id="apiKeyInput" required
                                class="flex-1 px-3 py-1.5 text-sm border rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                                placeholder="Enter your API key">
                            <button type="button" id="saveApiKey"
                                class="text-xs bg-gray-200 hover:bg-gray-300 px-3 py-1.5 rounded-lg transition">
                                Save
                            </button>
                        </div>
                        <div id="messages" class="messages space-y-3 mb-4 flex-1 overflow-y-auto">
                            <div class="flex justify-center">
                                <span class="text-xs text-gray-400">Enter your API key above to start</span>
                            </div>
                        </div>
                        <form id="chatForm" class="flex gap-2 mt-auto">
                            <textarea type="text" id="messageInput" rows="2"
                                class="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500 resize-none"
                                placeholder="Type your message..." required></textarea>
                            <button type="submit"
                                class="gradient-bg text-white px-6 py-2 rounded-lg hover:opacity-90 transition self-end">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </form>
                    </div>
                </div>

                <div class="mt-6 bg-white rounded-xl shadow p-6">
                    <h3 class="font-semibold mb-4">Features</h3>
                    <ul class="space-y-2 text-sm text-gray-600">
                        <li class="flex items-center gap-2"><i class="fas fa-check text-green-500"></i> OpenAI-compatible API</li>
                        <li class="flex items-center gap-2"><i class="fas fa-check text-green-500"></i> Supports Chichewa & English</li>
                        <li class="flex items-center gap-2"><i class="fas fa-check text-green-500"></i> RAG-ready with pgvector</li>
                        <li class="flex items-center gap-2"><i class="fas fa-check text-green-500"></i> GPU acceleration</li>
                    </ul>
                </div>
            </div>

            <div class="space-y-6">
                <div class="bg-orange-50 border border-orange-200 rounded-xl p-6">
                    <h3 class="font-semibold text-orange-900 mb-2">Build Your Own</h3>
                    <p class="text-sm text-orange-700 mb-4">Fork this template on GitHub, customize the AI behavior, and deploy your own assistant.</p>
                    <a href="{repo_url}" target="_blank" class="inline-flex items-center gap-2 bg-orange-600 text-white px-4 py-2 rounded-lg hover:bg-orange-700 transition text-sm">
                        <i class="fab fa-github"></i> View on GitHub
                    </a>
                </div>

                <div class="bg-white rounded-xl shadow p-6">
                    <h3 class="font-semibold mb-4 flex items-center gap-2"><i class="fas fa-terminal"></i> API Quick Start</h3>
                    <div class="space-y-3">
                        <div>
                            <p class="text-xs font-semibold text-gray-500 mb-1">Python</p>
                            <pre class="bg-gray-900 text-gray-100 p-3 rounded-lg text-xs overflow-x-auto">import httpx

response = httpx.post(
    "{api_base_url}/v1/chat/completions",
    headers={{"X-API-Key": "YOUR_API_KEY"}},
    json={{
        "model": "phi3:latest",
        "messages": [{{"role": "user", "content": "Hello"}}]
    }}
)
print(response.json())</pre>
                        </div>
                        <div>
                            <p class="text-xs font-semibold text-gray-500 mb-1">cURL</p>
                            <pre class="bg-gray-900 text-gray-100 p-3 rounded-lg text-xs overflow-x-auto">curl -X POST "{api_base_url}/v1/chat/completions" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{{"model": "phi3:latest", "messages": [{{"role": "user", "content": "Hello"}}]}}'</pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer class="mt-12 py-6 text-center text-gray-400 text-sm">
        <p>Powered by <span class="font-semibold">42helv</span> AI Infrastructure</p>
    </footer>

    <script>
        const API_BASE_URL = 'https://ai-api.42helv.com';
        const messagesDiv = document.getElementById('messages');
        const chatForm = document.getElementById('chatForm');
        const messageInput = document.getElementById('messageInput');
        const apiKeyInput = document.getElementById('apiKeyInput');
        const saveApiKeyBtn = document.getElementById('saveApiKey');

        const savedKey = localStorage.getItem('ai_api_key');
        if (savedKey) apiKeyInput.value = savedKey;

        saveApiKeyBtn.addEventListener('click', () => {{
            const key = apiKeyInput.value.trim();
            if (key) {{
                localStorage.setItem('ai_api_key', key);
                apiKeyInput.type = 'text';
                setTimeout(() => {{ apiKeyInput.type = 'password'; }}, 1000);
            }}
        }});

        function addMessage(content, isUser = false) {{
            const div = document.createElement('div');
            div.className = `flex ${{isUser ? 'justify-end' : 'justify-start'}}`;
            div.innerHTML = `<div class="${{isUser ? 'message-user' : 'message-ai'}} px-4 py-2 rounded-lg max-w-[80%] break-words">${{content.replace(/\\n/g, '<br>')}}</div>`;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }}

        function addTyping() {{
            const div = document.createElement('div');
            div.id = 'typing';
            div.className = 'flex justify-start';
            div.innerHTML = `<div class="message-ai px-4 py-2 rounded-lg"><div class="flex gap-1"><span class="typing-dot w-2 h-2 bg-gray-400 rounded-full"></span><span class="typing-dot w-2 h-2 bg-gray-400 rounded-full" style="animation-delay:.2s"></span><span class="typing-dot w-2 h-2 bg-gray-400 rounded-full" style="animation-delay:.4s"></span></div></div>`;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }}

        function removeTyping() {{
            const t = document.getElementById('typing');
            if (t) t.remove();
        }}

        chatForm.addEventListener('submit', async (e) => {{
            e.preventDefault();
            const msg = messageInput.value.trim();
            if (!msg) return;
            addMessage(msg, true);
            messageInput.value = '';
            addTyping();
            try {{
                const headers = {{ 'Content-Type': 'application/json' }};
                const apiKey = localStorage.getItem('ai_api_key') || '';
                headers['X-API-Key'] = apiKey;
                const res = await fetch(API_BASE_URL + '/v1/chat/completions', {{
                    method: 'POST',
                    headers,
                    body: JSON.stringify({{
                        model: 'phi3:latest',
                        messages: [{{'role': 'user', 'content': msg}}],
                        temperature: 0.7
                    }})
                }});
                const data = await res.json();
                removeTyping();
                if (data.choices && data.choices[0] && data.choices[0].message) {{
                    addMessage(data.choices[0].message.content);
                }} else if (data.error) {{
                    addMessage('Error: ' + data.error);
                }} else {{
                    addMessage('AI service unavailable: ' + JSON.stringify(data));
                }}
            }} catch (e) {{
                removeTyping();
                addMessage('Error connecting to AI service: ' + e.message);
            }}
        }});
    </script>
</body>
</html>
"""

WIDGET_JS = """
(function() {
    window.ChatWidget = {
        config: { apiUrl: '', title: 'AI Assistant', position: 'bottom-right' },
        init: function(opts) {
            this.config = { ...this.config, ...opts };
            this.injectStyles();
            this.createWidget();
            document.getElementById('chatWidgetForm').onsubmit = (e) => {
                e.preventDefault();
                const inp = document.getElementById('chatWidgetInput');
                const msg = inp.value.trim();
                if (!msg) return;
                this.addMessage(msg, true);
                inp.value = '';
                fetch(this.config.apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: 'phi3:latest', messages: [{ role: 'user', content: msg }] })
                }).then(r => r.json()).then(d => this.addMessage(d.choices[0].message.content)).catch(() => this.addMessage('Error'));
            };
        },
        injectStyles: function() {
            const s = document.createElement('style');
            s.textContent = `
                .chat-widget-btn { position: fixed; ${this.config.position}: 20px; bottom: 20px; width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #f97316, #ea580c); border: none; cursor: pointer; box-shadow: 0 4px 20px rgba(249,115,22,.4); z-index: 9999; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; }
                .chat-widget-window { position: fixed; ${this.config.position}: 20px; bottom: 90px; width: 380px; height: 500px; background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,.15); z-index: 9998; display: none; flex-direction: column; overflow: hidden; }
                .chat-widget-window.open { display: flex; }
                .chat-widget-header { background: linear-gradient(135deg, #f97316, #ea580c); color: white; padding: 16px; display: flex; align-items: center; gap: 12px; }
                .chat-widget-messages { flex: 1; overflow-y: auto; padding: 16px; }
                .chat-widget-input { display: flex; padding: 16px; gap: 8px; border-top: 1px solid #eee; }
                .chat-widget-input input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; }
                .chat-widget-input button { background: linear-gradient(135deg, #f97316, #ea580c); color: white; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; }
            `;
            document.head.appendChild(s);
        },
        createWidget: function() {
            document.body.insertAdjacentHTML('beforeend', `
                <button class="chat-widget-btn" id="chatWidgetBtn"><i class="fas fa-comments"></i></button>
                <div class="chat-widget-window" id="chatWidgetWindow">
                    <div class="chat-widget-header"><div style="width:40px;height:40px;background:rgba(255,255,255,.2);border-radius:50%;display:flex;align-items:center;justify-content:center;"><i class="fas fa-robot"></i></div><div><div style="font-weight:600">${this.config.title}</div><div style="font-size:12px;opacity:.8">AI Assistant</div></div></div>
                    <div class="chat-widget-messages" id="chatWidgetMessages"></div>
                    <form class="chat-widget-input" id="chatWidgetForm"><input type="text" placeholder="Type a message..." id="chatWidgetInput" /><button type="submit"><i class="fas fa-paper-plane"></i></button></form>
                </div>
            `);
            document.getElementById('chatWidgetBtn').onclick = () => document.getElementById('chatWidgetWindow').classList.toggle('open');
        },
        addMessage: function(content, isUser) {
            const d = document.createElement('div');
            d.style.cssText = `text-align:${isUser?'right':'left'};margin:8px 0;`;
            d.innerHTML = `<span style="display:inline-block;padding:10px 14px;border-radius:12px;background:${isUser?'#f97316':'#fff7ed'};color:${isUser?'white':'#9a3412'}">${content}</span>`;
            document.getElementById('chatWidgetMessages').appendChild(d);
            document.getElementById('chatWidgetMessages').scrollTop = document.getElementById('chatWidgetMessages').scrollHeight;
        }
    };
})();
"""

REPO_URL = "https://github.com/dannysimfukwe/healthcare-ai"

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    title = AI_TITLES.get(AI_TYPE, "Healthcare AI")
    html = LANDING_PAGE.format(
        title=title,
        repo_url=REPO_URL,
        base_url=BASE_URL,
        api_base_url=AI_API_BASE_URL
    )
    return HTMLResponse(content=html)

@app.get("/widget.js")
async def widget_js():
    return Response(content=WIDGET_JS, media_type="application/javascript")

@app.get("/health")
async def health():
    return {"status": "healthy", "ai_type": AI_TYPE}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except:
        return {"error": "Invalid JSON body"}

    messages = body.get("messages", [])
    model = body.get("model", "phi3:latest")
    temperature = body.get("temperature", 0.7)
    system_prompt = f"You are a {AI_TITLES.get(AI_TYPE, 'healthcare')} assistant."
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    api_key = request.headers.get("X-API-Key") or API_KEY
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{AI_API_URL}/chat/completions",
                headers=headers,
                json={"model": model, "messages": full_messages, "temperature": temperature}
            )
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    return data
                else:
                    return {"error": f"Invalid AI response (no choices): {str(data)[:200]}", "raw": data}
            else:
                return {"error": f"AI service error {response.status_code}", "detail": response.text[:500]}
    except httpx.ConnectError as e:
        return {"error": f"Connection failed to {AI_API_URL}. Is the service reachable?", "detail": str(e)}
    except httpx.TimeoutException:
        return {"error": "AI service timed out after 120s"}
    except Exception as e:
        return {"error": f"Unexpected error: {type(e).__name__}", "detail": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    api_key = request.headers.get("X-API-Key") or API_KEY
    messages = request.messages or []
    user_message = messages[-1]["content"] if messages else ""
    ollama_response = await call_ollama(user_message, "general", api_key)
    return {"response": ollama_response, "language": "en", "confidence": 0.85}

@app.post("/clinical-notes")
async def clinical_notes(text: str):
    structured = await generate_clinical_notes(text)
    return {"notes": structured}

@app.post("/translate")
async def translate(text: str, from_lang: str = "en", to_lang: str = "ny"):
    translated = await translate_text(text, from_lang, to_lang)
    return {"translated": translated, "from": from_lang, "to": to_lang}

async def call_ollama(prompt: str, context: str, api_key: str = None) -> str:
    system_prompt = f"You are a {AI_TITLES.get(AI_TYPE, 'healthcare')} assistant. Context: {context}. Provide accurate, helpful information."
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{AI_API_URL}/chat/completions",
                headers=headers,
                json={
                    "model": "phi3:latest",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
            )
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                elif "error" in data:
                    return f"AI error: {data['error']}"
            return f"AI service returned status {response.status_code}: {response.text}"
    except httpx.ConnectError as e:
        return f"AI service unavailable. Could not connect. Error: {str(e)}"
    except httpx.TimeoutException:
        return "AI service unavailable. Request timed out."
    except Exception as e:
        return f"AI service unavailable. Error: {str(e)}"

async def generate_clinical_notes(text: str) -> dict:
    notes = await call_ollama(f"Convert this dictation into structured clinical notes: {text}", "clinical documentation")
    return {"chief_complaint": "Extracted", "history": notes, "assessment": "AI-generated", "plan": "Pending review"}

async def translate_text(text: str, from_lang: str, to_lang: str) -> str:
    lang_names = {"en": "English", "ny": "Chichewa", "tob": "Tumbuka"}
    prompt = f"Translate from {lang_names.get(from_lang, from_lang)} to {lang_names.get(to_lang, to_lang)}: {text}"
    return await call_ollama(prompt, "translation")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
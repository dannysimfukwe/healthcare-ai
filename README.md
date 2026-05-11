# Healthcare AI Template for 42helv

RAG-powered medical assistant with Chichewa support.

## Features
- OpenAI-compatible API endpoint (`/v1/chat/completions`)
- Clinical notes generation from dictation
- English ↔ Chichewa translation
- RAG pipeline with pgvector
- GPU acceleration

## API Endpoints

### Chat
```bash
POST /v1/chat/completions
{
  "message": "Patient has fever and cough",
  "language": "en",
  "context": "general"
}
```

### Clinical Notes
```bash
POST /clinical-notes
Body: "Patient complains of fever since 3 days..."
```

### Translation
```bash
POST /translate
Body: "Hello, how are you?", from_lang: "en", to_lang: "ny"
```

## Environment
- `OLLAMA_BASE_URL` - Ollama service URL (default: http://ollama:11434)
- `DB_HOST` - PostgreSQL host for pgvector
- `RAG_MODE` - Enable RAG pipeline (true/false)

## Setup

1. Deploy Ollama AI template first
2. Deploy PostgreSQL with pgvector extension
3. Deploy Healthcare AI with connection to both
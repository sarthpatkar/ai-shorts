# AI Shorts Generator

🚀 Turn long-form videos into viral, high-retention short-form clips using an end-to-end AI pipeline.

> AI-powered system that ingests videos and automatically generates short-form content optimized for engagement (Reels / Shorts / TikTok format)

---

## 🎬 Demo

https://your-demo-link-here.com  
*(Add a 30–60 sec Loom or screen recording here — this is CRITICAL)*

---

## 🖼️ Preview

![App Screenshot](https://via.placeholder.com/1200x600.png?text=AI+Shorts+Generator+Preview)

---

## ⚡ Key Highlights

- 🎯 Converts long videos → multiple short viral clips
- ⚙️ Full pipeline: ingestion → transcription → scoring → clipping
- 🧠 AI-driven segment selection
- 🔁 Async job system with status tracking
- 🔒 Secure delivery via signed URLs
- 🚀 Built with production-ready architecture

---

## 🏗️ Architecture

```
Frontend (Next.js)
        ↓
API Routes
        ↓
Backend (FastAPI)
        ↓
Async Job Pipeline
        ↓
yt-dlp + ffmpeg + transcription
        ↓
Supabase Storage
        ↓
Signed URLs → Frontend
```

---

## ⚙️ Tech Stack

### Frontend
- Next.js (App Router)
- React + TypeScript
- Tailwind CSS

### Backend
- FastAPI (Python)
- ffmpeg
- yt-dlp

### Infra
- Supabase (DB + Storage)
- Signed URL delivery

---

## 🧠 System Design (What makes this strong)

### 1. Stage-Based Pipeline
Each step is isolated → better debugging, retry, scalability.

### 2. Async Job Execution
Heavy processing happens in background → fast UI.

### 3. Partial Success Handling
Even if some clips fail → others still delivered.

### 4. Secure Media Access
No public file exposure → signed URLs only.

---

## 📦 Features

- YouTube ingestion (yt-dlp)
- Direct video uploads
- Multi-clip generation
- Feedback system
- Download options with gating
- Rate limiting (anti-abuse)
- Internal API protection

---

## 🧪 Local Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- ffmpeg
- yt-dlp
- Supabase project

---

### Setup

```bash
git clone https://github.com/sarthpatkar/ai-shorts.git
cd ai-shorts
```

#### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🔐 Security

- No secrets committed
- Internal API token protection
- Rate limiting on heavy endpoints
- Signed URLs for media

---

## ⚠️ Limitations

- Processing time depends on system resources
- YouTube anti-bot restrictions may affect downloads
- Local rate limiting is not distributed

---

## 🔮 Future Work

- Distributed workers (Celery / Redis)
- Better clip ranking models
- Global rate limiting
- Observability (logs, metrics)

---

## 📄 License
MIT

---

## 👨‍💻 Author

**Sarth Patkar**

---

## ⭐ If you found this interesting

Give it a star — it helps visibility and motivates further development.
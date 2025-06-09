from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
from app.pipeline import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html",encoding='utf-8') as f:
        return f.read()

@app.post("/summarize/")
async def summarize_video(
    video: UploadFile,
    format_type: str = Form(...),
    length: str = Form(...)
):
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    
    with open(video_path, "wb") as f:
        f.write(await video.read())

    transcript, summary = run_pipeline(video_path, format_type, length)

    return {"transcript": transcript, "summary": summary}

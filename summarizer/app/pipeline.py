import os
import whisper
import torch
import textwrap
import warnings
import contextlib
import io
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
model_whisper = whisper.load_model("small").to("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model_summarizer = model_summarizer.half()

def extract_audio(video_path, audio_path):
    import subprocess
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def transcribe_audio(audio_path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = model_whisper.transcribe(audio_path)
    return result["text"]

def summarize_text(text, format_type, length):
    chunks = textwrap.wrap(text, width=1024)
    
    def summarize_chunk(chunk):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(model_summarizer.device)
        with torch.no_grad():
            summary_ids = model_summarizer.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=30,
                num_beams=4,
                early_stopping=True
            )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize_chunk, chunks))

    final_summary = " ".join(summaries)
    
    if length == "short":
        sentences = final_summary.split(". ")
        final_summary = ". ".join(sentences[:2]) + "."
    elif length == "medium":
        sentences = final_summary.split(". ")
        final_summary = ". ".join(sentences[:4]) + "."
    
    if format_type == "bullet points":
        bullets = final_summary.split(". ")
        final_summary = "\n• " + "\n• ".join([b.strip() for b in bullets if b.strip()]) + "."

    return final_summary

def run_pipeline(video_path, format_type, length):
    audio_path = video_path.replace(".mp4", ".mp3")
    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)
    summary = summarize_text(transcript, format_type, length)
    return transcript, summary

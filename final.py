import subprocess
import whisper
import contextlib
import io
import warnings
import textwrap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==== AUDIO EXTRACTION ====
def extract_audio(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ Audio extracted to: {audio_path}")
    except subprocess.CalledProcessError:
        print("❌ FFmpeg extraction failed.")

# ==== TRANSCRIPTION USING GPU ====
def transcribe_audio(audio_path, progress_bar):
    print("🔍 Transcribing audio into text...")
    model = whisper.load_model("small").to("cuda" if torch.cuda.is_available() else "cpu")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = model.transcribe(audio_path)
    print("✅ Transcription complete.")
    progress_bar.update(1)
    return result["text"]

# ==== FORMAT & LENGTH HANDLING ====
def adjust_length(summary, level):
    sentences = summary.split(". ")
    if level == "short":
        return ". ".join(sentences[:2]) + "."
    elif level == "medium":
        return ". ".join(sentences[:4]) + "."
    else:  # long
        return summary

def format_output(text, format_type):
    if format_type == "bullet points":
        bullets = text.split(". ")
        return "\n• " + "\n• ".join([b.strip() for b in bullets if b.strip()]) + "."
    return text  # Paragraphs (default)

# ==== SUMMARIZATION ====
def summarize_text(text, progress_bar, format_type, length):
    print("📚 Summarizing...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if device.type == 'cuda':
        model = model.half()

    chunks = textwrap.wrap(text, width=1024)
    print(f"🧩 {len(chunks)} chunk(s) to summarize...")

    progress_bar.total += len(chunks)
    progress_bar.refresh()

    def summarize_chunk(chunk):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=30,
                num_beams=4,
                early_stopping=True
            )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    summaries = []
    with ThreadPoolExecutor() as executor:
        for summary in tqdm(executor.map(summarize_chunk, chunks), total=len(chunks), desc="✨ Summarizing", leave=False):
            summaries.append(summary)
            progress_bar.update(1)

    final_summary = " ".join(summaries)
    final_summary = adjust_length(final_summary, length)
    final_summary = format_output(final_summary, format_type)

    return final_summary

# ==== MAIN PIPELINE ====
if __name__ == "__main__":
    video_path = input("🎥 Enter video file path: ").strip('"')

    format_type = input("📝 Summary Format (Paragraphs / Bullet Points): ").strip().lower()
    length = input("📏 Summary Length (Short / Medium / Long): ").strip().lower()

    audio_path = os.path.join(script_dir, "extracted_audio.mp3")
    transcript_path = os.path.join(script_dir, "transcript.txt")
    summary_path = os.path.join(script_dir, "summary.txt")

    progress_bar = tqdm(total=2, desc="Overall Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}]")

    extract_audio(video_path, audio_path)
    progress_bar.update(1)

    text = transcribe_audio(audio_path, progress_bar)

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)

    summary = summarize_text(text, progress_bar, format_type, length)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    progress_bar.close()

    print("\n📝 Final Summary:\n")
    print(summary)

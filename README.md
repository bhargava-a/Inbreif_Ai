# Inbreif_Ai - Video Summarizer 
This Python project converts any video file into a summarized text using OpenAI's Whisper for audio transcription and Facebook's BART for text summarization.

Features:
  Extracts audio from video using FFmpeg

  Transcribes speech using Whisper (small model)

  Summarizes content using BART (facebook/bart-large-cnn)

Custom summary format:

  Paragraphs or Bullet Points

  Short, Medium, or Long summaries

Runs on CPU or GPU automatically

Generates:

  transcript.txt — full transcription

  summary.txt — summarized output


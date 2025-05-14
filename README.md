<h1 align="center">Inbrief_Ai - Video Summarizer </h1>
<p align="center"><i>Transcribe any video and summarize it intelligently using OpenAI Whisper + Facebook BART</i></p>
<hr>

<h2> Overview</h2>
<p>This tool extracts speech from videos, converts it into text using <b>OpenAI Whisper</b>, and summarizes the text using <b>Facebook's BART Transformer</b>.</p>
<ul>
  <li>⚙️ Runs on CPU or GPU</li>
  <li>📊 Real-time progress with <code>tqdm</code></li>
</ul>

<hr>

<h2> Features</h2>
<ul>
  <li>🎧 Audio extraction from videos using <code>FFmpeg</code></li>
  <li>🗣️ Accurate speech-to-text transcription via Whisper</li>
  <li>🧠 Smart summarization using BART</li>
  <li>📝 Summary format options: <b>Paragraphs</b> or <b>Bullet Points</b></li>
  <li>📏 Summary length choices: <b>Short</b>, <b>Medium</b>, or <b>Long</b></li>
  <li>💾 Outputs: <code>transcript.txt</code> and <code>summary.txt</code></li>
</ul>

<hr>

<h2>📦 Requirements</h2>

<h4>✅ Python Libraries</h4>
<pre><code>pip install -r requirements.txt</code></pre>

<pre><code>
whisper
transformers
torch
tqdm
</code></pre>

<h4>🔧 FFmpeg</h4>
<p>Download from <a href="https://ffmpeg.org/download.html" target="_blank">ffmpeg.org</a> and add it to your system <code>PATH</code>.</p>

<hr>

<h2>💻 How to Run</h2>
<pre><code>python main.py</code></pre>

<h4>Example Prompts</h4>
<pre>
🎥 Enter video file path: example.mp4
📝 Summary Format (Paragraphs / Bullet Points): bullet points
📏 Summary Length (Short / Medium / Long): medium
</pre>

<hr>

<h2>📂 Output Files</h2>
<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>extracted_audio.mp3</code></td>
    <td>Audio extracted from video</td>
  </tr>
  <tr>
    <td><code>transcript.txt</code></td>
    <td>Full transcription of audio</td>
  </tr>
  <tr>
    <td><code>summary.txt</code></td>
    <td>Summarized version of transcript</td>
  </tr>
</table>

<hr>

<h2> Tech Stack</h2>
<ul>
  <li><a href="https://github.com/openai/whisper" target="_blank">Whisper (OpenAI)</a></li>
  <li><a href="https://huggingface.co/facebook/bart-large-cnn" target="_blank">Transformers – facebook/bart-large-cnn</a></li>
  <li><a href="https://pytorch.org/" target="_blank">PyTorch</a></li>
  <li><a href="https://ffmpeg.org/" target="_blank">FFmpeg</a></li>
</ul>

<hr>

<h2>📁 Folder Structure</h2>
<pre>
project-root/
├── main.py
├── extracted_audio.mp3     # Auto-generated
├── transcript.txt          # Auto-generated
├── summary.txt             # Auto-generated
└── requirements.txt
</pre>

<hr>


🔗 <a href="https://github.com/bhargava-a/Inbreif_Ai" target="_blank">GitHub</a> 

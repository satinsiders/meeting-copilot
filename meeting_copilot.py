# file: meeting_assistant.py
import os, io, queue, threading, time, sys, tempfile, subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np

from pynput import keyboard
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# pip install sounddevice soundfile numpy pynput python-dotenv openai
# Requires ffmpeg or playsound for audio playback on Linux
# On macOS, uses built-in afplay for audio playback

# ====== OpenAI Client Setup ======
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== File Paths ======
TRANSCRIPTS_DIR = "transcripts"
TRANSCRIPT_PATH = None  # set at startup per run

# ====== Settings ======
listener = None
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 6               # rolling dictation chunk length
MODEL_STT = "gpt-4o-mini-transcribe"
MODEL_LLM = "gpt-5-mini"
MODEL_TTS = "gpt-4o-mini-tts"
VOICE = "alloy"                 # try: verse, aria, breeze, etc.
INCLUDE_LAST_SECONDS = 900      # when sending: ~15 min window
SYSTEM_PROMPT = """You are a concise, helpful meeting copilot.
- Read the transcript context.
- Participate as one of the attendees and respond in character.
- Speak in natural, conversational language using complete sentences.
- Avoid lists, bullet points, or numbering.
- Keep replies brief—one or two sentences.
- Offer useful insights, suggestions, or action items.
- If uncertain, ask one clarifying question at most.
"""

# ====== Globals ======
audio_q = queue.Queue()
stop_capture = threading.Event()
stop_program = threading.Event()
SEND_FLAG = threading.Event()


def new_transcript_path():
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(TRANSCRIPTS_DIR, f"transcript_{ts}.md")

def ensure_file(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Live Meeting Transcript\n\n")

def append_transcript(text):
    if not TRANSCRIPT_PATH:
        return
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def read_recent_transcript(seconds=INCLUDE_LAST_SECONDS):
    if not TRANSCRIPT_PATH or not os.path.exists(TRANSCRIPT_PATH):
        return ""
    try:
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-200:])
    except FileNotFoundError:
        return ""


def record_stream():
    """Continuously capture mic audio and put CHUNK_SECONDS buffers into audio_q."""
    while not stop_capture.is_set():
        chunk_frames = []
        def callback(indata, frames, t, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            # Append the whole frame (copy) as one block
            chunk_frames.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=callback):
            start = time.time()
            while (time.time() - start) < CHUNK_SECONDS and not stop_capture.is_set():
                sd.sleep(50)

        if not chunk_frames:
            continue

        # Concatenate all captured frames in this chunk
        np_audio = np.concatenate(chunk_frames, axis=0)

        # Write WAV into an in-memory buffer
        buf = io.BytesIO()
        sf.write(buf, np_audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        buf.seek(0)

        audio_q.put(buf)


def stt_worker():
    """Read audio chunks, transcribe, and append to transcript."""
    while not stop_program.is_set():
        try:
            buf = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            # OpenAI Whisper transcription
            tr = client.audio.transcriptions.create(
                model=MODEL_STT,
                file=("chunk.wav", buf, "audio/wav"),
                response_format="text"
            )
            text = tr.strip()
            if text:
                append_transcript(text)
                print(f"[STT] {text}")
        except Exception as e:
            print(f"[STT ERROR] {e}", file=sys.stderr)


def synthesize_and_play(text):
    try:
        # Request speech (MP3 by default; no unsupported 'format' kwarg)
        speech = client.audio.speech.create(
            model=MODEL_TTS,   # e.g., "gpt-4o-mini-tts"
            voice=VOICE,       # e.g., "alloy"
            input=text,
        )

        audio_bytes = speech.read()

        # Write to a temp MP3 and play it (blocking) so it "speaks now"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        if sys.platform == "darwin":                # macOS
            subprocess.run(["afplay", tmp_path], check=False)
        elif sys.platform.startswith("win"):        # Windows
            os.startfile(tmp_path)                  # plays via default app
        else:                                       # Linux / other
            try:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path], check=False)
            except FileNotFoundError:
                from playsound import playsound     # pip install playsound
                playsound(tmp_path)

        os.unlink(tmp_path)
    except Exception as e:
        print(f"[TTS ERROR] {e}", file=sys.stderr)


def llm_respond():
    context = read_recent_transcript()
    if not context.strip():
        print("[LLM] No transcript yet.")
        return
    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            instructions=SYSTEM_PROMPT,
            input=[{ "role":"user", "content":[{"type":"input_text","text": context}] }],
            reasoning={"effort": "low"},      # reduce hidden reasoning spend
            text={"verbosity": "low"},
            max_output_tokens=80,
        )
        reply = (getattr(resp, "output_text", None) or "").strip()
        if not reply:
            print("[LLM] Empty reply (no text to speak). Full response object below for debugging:")
            print(resp)
            return

        print(f"\n[LLM]\n{reply}\n")
        synthesize_and_play(reply)

    except Exception as e:
        print(f"[LLM ERROR] {e}", file=sys.stderr)


def shutdown(reason="[shutdown]"):
    print(f"\n{reason} Cleaning up...")
    stop_program.set()
    stop_capture.set()
    try:
        if listener is not None:
            listener.stop()
            listener.join()
        sd.stop()
    except Exception:
        pass

def on_press(key):
    try:
        if key == keyboard.Key.enter:
            SEND_FLAG.set()
        elif key == keyboard.Key.esc:
            stop_program.set()
            return False
    except Exception:
        pass


def main():
    global listener, TRANSCRIPT_PATH
    
    print("Live Meeting Assistant (local, in-person)")
    print("• Always-on dictation → transcript.md")
    print("• Press ENTER to make the assistant speak.")
    print("• Press ESC or Ctrl-C to quit.\n")

    # Prepare transcript file
    TRANSCRIPT_PATH = new_transcript_path()
    ensure_file(TRANSCRIPT_PATH)
    print(f"[init] Writing transcript to: {TRANSCRIPT_PATH}")

    # Start capture + STT threads
    t_capture = threading.Thread(target=record_stream, daemon=True)
    t_stt = threading.Thread(target=stt_worker, daemon=True)
    t_capture.start()
    t_stt.start()

    # Start keyboard listener for keypress detection
    listener = keyboard.Listener(on_press=on_press, daemon=True)
    listener.start()

    try:
        while not stop_program.is_set():
            if SEND_FLAG.is_set():
                SEND_FLAG.clear()
                llm_respond()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown("[exit]")
        t_capture.join(timeout=1.0)
        t_stt.join(timeout=1.0)
        print("Bye.")


if __name__ == "__main__":
    main()

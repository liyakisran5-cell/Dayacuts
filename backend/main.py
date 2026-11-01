import os, re, uuid, shutil, subprocess, threading
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE = Path("/tmp/dayaedits")
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
JOBS = {}
LOCK = threading.Lock()

app = FastAPI(title="DayaEdits Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-5000:])
    return p

def has_ffmpeg():
    return shutil.which("ffmpeg") and shutil.which("ffprobe")

def duration_of(path):
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(p.stdout.strip())

def make_srt(video_path, srt_path, language="auto"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("Auto captions need faster-whisper. Install requirements.txt first.")
    # CPU/int8 keeps the app usable on machines without an NVIDIA GPU.
    model = WhisperModel(os.getenv("WHISPER_MODEL", "small"),
                         device=os.getenv("WHISPER_DEVICE", "cpu"),
                         compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    lang = None if language == "auto" else language
    segments, info = model.transcribe(str(video_path), language=lang, vad_filter=True)
    def stamp(seconds):
        ms = int(seconds * 1000)
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i, seg in enumerate(segments, 1):
        txt = re.sub(r"\s+", " ", seg.text).strip()
        if txt:
            lines += [str(i), f"{stamp(seg.start)} --> {stamp(seg.end)}", txt, ""]
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines) // 4

def process_job(job_id, source, opts):
    try:
        with LOCK:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["progress"] = 5
            JOBS[job_id]["message"] = "Preparing video…"
        if not has_ffmpeg():
            raise RuntimeError("FFmpeg/FFprobe not found. Install FFmpeg and add it to PATH.")
        src = Path(source)
        out = OUTPUTS / f"{job_id}_edited.mp4"
        work = UPLOADS / f"{job_id}_work.mp4"
        current = src

        filters = []
        # Color correction
        grade = opts.get("grade", "original")
        grades = {
            "cinematic": "eq=contrast=1.12:saturation=1.16:brightness=-0.015,colorbalance=rs=.04:gs=.01:bs=-.03",
            "warm": "eq=contrast=1.06:saturation=1.18,colorbalance=rs=.08:gs=.02:bs=-.06",
            "cool": "eq=contrast=1.08:saturation=1.04,colorbalance=rs=-.04:gs=.01:bs=.07",
            "vivid": "eq=contrast=1.08:saturation=1.38",
            "bw": "hue=s=0,eq=contrast=1.12"
        }
        vf = []
        if grade in grades: vf.append(grades[grade])
        if opts.get("vertical"):
            vf.append("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920")
        if opts.get("fade"):
            dur = max(0.1, duration_of(current))
            vf += [f"fade=t=in:st=0:d=0.5", f"fade=t=out:st={max(0,dur-0.6):.3f}:d=0.6"]
        speed = float(opts.get("speed", 1.0))
        speed = min(2.0, max(0.5, speed))
        af = []
        if opts.get("denoise"): af.append("afftdn=nf=-25")
        if speed != 1.0:
            vf.append(f"setpts={1/speed:.6f}*PTS")
            # atempo accepts 0.5..2.0
            af.append(f"atempo={speed:.4f}")
        if opts.get("remove_silence"):
            # FFmpeg's silenceremove trims long quiet sections from the audio track.
            # It keeps the video timeline synchronized using the matching video filter.
            # Use conservative thresholds to avoid cutting normal speech pauses.
            af.insert(0, "silenceremove=start_periods=1:start_duration=0.45:start_threshold=-42dB:stop_periods=-1:stop_duration=0.8:stop_threshold=-42dB")
        if vf or af:
            cmd = ["ffmpeg", "-y", "-i", str(current)]
            if vf: cmd += ["-vf", ",".join(vf)]
            if af: cmd += ["-af", ",".join(af)]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", str(work)]
            run(cmd)
            current = work

        with LOCK:
            JOBS[job_id]["progress"] = 45
            JOBS[job_id]["message"] = "Applying captions and finishing video…"

        if opts.get("captions"):
            srt = UPLOADS / f"{job_id}.srt"
            count = make_srt(current, srt, opts.get("language", "auto"))
            # FFmpeg subtitles filter burns captions into the exported video.
            escaped = str(srt.resolve()).replace("\\", "/").replace(":", r"\:")
            capout = OUTPUTS / f"{job_id}_captioned.mp4"
            style = "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=55,Bold=1"
            run(["ffmpeg", "-y", "-i", str(current), "-vf",
                 f"subtitles='{escaped}':force_style='{style}'",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-c:a", "copy", str(capout)])
            current = capout
            with LOCK: JOBS[job_id]["caption_count"] = count

        if current != out:
            shutil.copy2(current, out)
        with LOCK:
            JOBS[job_id].update(status="completed", progress=100, message="Your edited video is ready.",
                                download_url=f"/api/download/{job_id}", filename=out.name)
    except Exception as e:
        with LOCK:
            JOBS[job_id].update(status="failed", message=str(e), error=str(e))
    finally:
        for p in [UPLOADS / f"{job_id}_work.mp4", UPLOADS / f"{job_id}.srt",
                  OUTPUTS / f"{job_id}_captioned.mp4"]:
            try: p.unlink(missing_ok=True)
            except Exception: pass

@app.get("/api/health")
def health():
    return {"ok": True, "ffmpeg": bool(has_ffmpeg()), "whisper_available": __import__("importlib").util.find_spec("faster_whisper") is not None}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename or not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video file.")
    ext = Path(file.filename).suffix.lower() or ".mp4"
    if ext not in {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}:
        raise HTTPException(400, "Unsupported video format.")
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{job_id}{ext}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 1024 * 1024 * 1024:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "Maximum upload size is 1 GB.")
            f.write(chunk)
    try: dur = duration_of(dest) if has_ffmpeg() else None
    except Exception: dur = None
    with LOCK:
        JOBS[job_id] = {"id": job_id, "status": "uploaded", "progress": 0,
                        "message": "Upload complete. Choose your edits.", "original_name": file.filename,
                        "duration": dur, "source": str(dest)}
    return {k:v for k,v in JOBS[job_id].items() if k != "source"}

@app.post("/api/process/{job_id}")
def process(job_id: str, background: BackgroundTasks,
            captions: bool = Form(False), remove_silence: bool = Form(False),
            grade: str = Form("original"), denoise: bool = Form(False),
            vertical: bool = Form(False), fade: bool = Form(False),
            speed: float = Form(1.0), language: str = Form("auto")):
    with LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, "Video upload not found.")
        if job["status"] == "processing": raise HTTPException(409, "This video is already processing.")
        if job["status"] == "completed": raise HTTPException(409, "This video has already been processed.")
        opts = {"captions":captions, "remove_silence":remove_silence, "grade":grade,
                "denoise":denoise, "vertical":vertical, "fade":fade,
                "speed":speed, "language":language}
        JOBS[job_id]["status"] = "queued"
    background.add_task(process_job, job_id, job["source"], opts)
    return {"id":job_id, "status":"queued", "message":"Your edit is queued."}

@app.get("/api/status/{job_id}")
def status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, "Job not found.")
        return {k:v for k,v in job.items() if k not in {"source"}}

@app.get("/api/download/{job_id}")
def download(job_id: str):
    from fastapi.responses import FileResponse
    path = OUTPUTS / f"{job_id}_edited.mp4"
    if not path.exists(): raise HTTPException(404, "Edited video is not ready.")
    return FileResponse(path, media_type="video/mp4", filename="DayaEdits_Export.mp4")

app.mount("/", StaticFiles(directory=str(BASE.parent / "frontend"), html=True), name="frontend")

"""세로 쇼츠 자동 편집기.

구글 플로우 등에서 받은 장면 영상(S1.mp4 ...)과 episode.json(대본·그래픽·효과음)을 받아
TTS 내레이션, 자막, 빨간 그래픽, 효과음을 얹은 1080x1920 완성본 mp4를 만든다.

    pip install -r shorts/requirements.txt
    python shorts/make_short.py shorts/interior/001_jolly-cut/episode.json --clips "C:/Users/me/Downloads/scenes"

장면 파일 이름이 S1, S2 ... 가 아니면 이름 순서대로 S1부터 짝지어 쓰고, 그 짝을 화면에 출력한다.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 48000
RED = (226, 40, 40, 255)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv"}
FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf",                       # 맑은 고딕 Bold (한국어 윈도우 기본)
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicExtraBold.ttf",
    str(Path(__file__).resolve().parent.parent / "fonts" / "Hahmlet-VF.ttf"),
]


def run(args):
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ffmpeg 실패:\n{r.stderr}")


def media_duration(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        sys.exit(f"길이를 읽을 수 없음: {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def pick_font(path=None):
    for p in [path, *FONT_CANDIDATES]:
        if p and Path(p).exists():
            return p
    sys.exit("한글 굵은 글꼴을 찾지 못했습니다. --font 로 .ttf 경로를 지정하세요.")


# ---------------------------------------------------------------- 장면 파일 찾기

def match_clips(clip_dir, scene_ids, order="name"):
    scene_ids = list(dict.fromkeys(scene_ids))  # S11처럼 앞 장면을 다시 쓰는 경우 중복 제거
    files = sorted(p for p in Path(clip_dir).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not files:
        sys.exit(f"영상 파일이 없습니다: {clip_dir}")
    by_stem = {p.stem.upper(): p for p in files}
    if all(sid.upper() in by_stem for sid in scene_ids):
        return {sid: by_stem[sid.upper()] for sid in scene_ids}

    def natural(p):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())]

    files.sort(key=(lambda p: p.stat().st_mtime) if order == "mtime" else natural)
    if len(files) < len(scene_ids):
        print(f"주의: 장면 {len(scene_ids)}개에 영상 {len(files)}개 — 모자란 장면은 앞 영상을 다시 씁니다.")
    mapping = {sid: files[i % len(files)] for i, sid in enumerate(scene_ids)}
    how = "다운로드(수정) 시각" if order == "mtime" else "이름"
    print(f"파일 이름이 S1, S2... 가 아니라서 {how} 순서대로 짝지었습니다:")
    for sid, p in mapping.items():
        print(f"  {sid:>4} <- {p.name}")
    return mapping


# ---------------------------------------------------------------- 내레이션

async def _edge_tts(text, voice, rate, out):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate).save(str(out))


def load_wav(path):
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768


def to_wav(src, dst, speed=1.0):
    af = ["-af", f"atempo={speed}"] if speed != 1.0 else []
    run(["-i", src, *af, "-ac", "1", "-ar", SR, "-sample_fmt", "s16", dst])
    return load_wav(dst)


_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 로컬 앱이라 프록시를 거치지 않음


def _http(method, url, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with _LOCAL.open(req, timeout=timeout) as r:
        return r.headers.get("Content-Type", ""), r.read()


def _is_audio(ctype, blob):
    return ctype.startswith("audio/") or blob[:4] in (b"RIFF", b"OggS", b"fLaC", b"ID3\x03", b"ID3\x04") \
        or blob[:2] == b"\xff\xfb"


def _voicebox(text, vb, out):
    """Voicebox 로컬 API: POST /generate → 작업 id → GET /audio/{id} 가 준비될 때까지 기다린다."""
    base = vb["url"].rstrip("/")
    body = {"text": text, "profile_id": vb["profile_id"], "language": vb.get("language", "ko"),
            **{k: v for k, v in vb.items() if k not in ("url", "profile_id", "language")}}
    ctype, blob = _http("POST", f"{base}/generate", body)
    if not _is_audio(ctype, blob):
        info = json.loads(blob)
        gid = info.get("id") or info.get("generation_id")
        if not gid:
            sys.exit(f"Voicebox 응답에서 작업 id를 찾지 못했습니다: {info}")
        for _ in range(600):
            try:
                ctype, blob = _http("GET", f"{base}/audio/{urllib.parse.quote(str(gid))}")
                if _is_audio(ctype, blob):
                    break
            except urllib.error.HTTPError as e:
                if e.code not in (404, 409, 425, 202):
                    raise
            time.sleep(1)
        else:
            sys.exit(f"Voicebox 음성이 10분 안에 준비되지 않았습니다: {text}")
    out.write_bytes(blob)


def voicebox_profiles(url):
    _, blob = _http("GET", url.rstrip("/") + "/profiles", timeout=10)
    profiles = json.loads(blob)
    profiles = profiles.get("profiles", profiles) if isinstance(profiles, dict) else profiles
    return [(p.get("id"), p.get("name")) for p in profiles]


def narrate(lines, cfg, work, mode, voice_dir=None):
    """줄마다 음성 배열을 돌려준다.
    edge=마이크로소프트 무료 음성, voicebox=PC의 Voicebox 앱, files=직접 만든 L000.wav ... 파일,
    silent=글자 수로 길이를 추정한 무음(화면 확인용)."""
    voices = []
    for i, text in enumerate(lines):
        if mode == "silent":
            syllables = len(re.sub(r"[^가-힣A-Za-z0-9]", "", text))
            dur = 0.3 + syllables * 0.11 + text.count(".") * 0.12
            voices.append(np.zeros(int(dur * SR), np.float32))
            continue
        if mode == "files":
            found = [p for p in Path(voice_dir).glob(f"L{i:03d}.*") if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")]
            if not found:
                sys.exit(f"음성 파일이 없습니다: {voice_dir}/L{i:03d}.wav  ({text})")
            voices.append(to_wav(found[0], work / f"files_L{i:03d}.wav", cfg["speed"]))
            continue
        # 대사·목소리 설정이 바뀌면 캐시도 새로 만든다
        key = hashlib.sha1(json.dumps([mode, text, cfg["voice"], cfg["rate"], cfg.get("voicebox")],
                                      ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]
        audio = work / f"{mode}_L{i:03d}_{key}.audio"
        if not audio.exists() or audio.stat().st_size == 0:
            # 임시 파일에 받은 뒤 옮겨서, 실패해도 빈 파일이 캐시에 남지 않게 한다
            tmp = audio.with_suffix(".part")
            for attempt in range(3):
                try:
                    if mode == "voicebox":
                        _voicebox(text, cfg["voicebox"], tmp)
                    else:
                        asyncio.run(_edge_tts(text, cfg["voice"], cfg["rate"], tmp))
                    if tmp.exists() and tmp.stat().st_size > 0:
                        tmp.replace(audio)
                        break
                except Exception as e:  # 네트워크 오류, 앱 꺼짐 등
                    print(f"  음성 생성 재시도 {attempt + 1}/3 ({text[:12]}…): {e}")
            else:
                tmp.unlink(missing_ok=True)
                hint = "Voicebox 앱이 켜져 있는지 확인하세요." if mode == "voicebox" else "인터넷 연결을 확인하세요."
                sys.exit(f"음성 생성 실패: {text}\n{hint} 화면만 먼저 보려면 --tts silent")
        speed = cfg["speed"] if mode == "voicebox" else 1.0
        voices.append(to_wav(audio, work / f"{mode}_L{i:03d}.wav", speed))
    return voices


# ---------------------------------------------------------------- 효과음 (파일이 없으면 합성)

def synth_sfx(name):
    t = lambda d: np.arange(int(d * SR)) / SR
    noise = lambda d: np.random.default_rng(7).uniform(-1, 1, int(d * SR))
    if name == "ting":
        x = t(0.9)
        return (np.sin(2 * np.pi * 2200 * x) + 0.4 * np.sin(2 * np.pi * 5300 * x)) * np.exp(-x * 6) * 0.35
    if name == "swoosh":
        x, n = t(0.45), noise(0.45)
        env = np.sin(np.pi * x / 0.45) ** 2
        return np.convolve(n, np.ones(12) / 12, "same") * env * 0.5
    if name == "impact":
        x = t(0.8)
        return (np.sin(2 * np.pi * (60 - 20 * x) * x) * np.exp(-x * 5) * 0.9
                + noise(0.8) * np.exp(-x * 60) * 0.3)
    if name == "click":
        x = t(0.06)
        return noise(0.06) * np.exp(-x * 120) * 0.6
    if name == "tick":
        x = t(0.05)
        return np.sin(2 * np.pi * 3000 * x) * np.exp(-x * 150) * 0.4
    if name == "grinder":
        x, n = t(0.5), noise(0.5)
        return (n * 0.3 + 0.2 * np.sign(np.sin(2 * np.pi * 180 * x))) * np.minimum(1, (0.5 - x) * 10) * 0.35
    if name == "crack":
        x = t(0.25)
        return noise(0.25) * np.exp(-x * 30) * 0.7
    if name == "clock":
        one = synth_sfx("tick")
        out = np.zeros(int(0.9 * SR), np.float32)
        for k in range(3):
            s = int(k * 0.3 * SR)
            out[s:s + len(one)] += one
        return out
    sys.exit(f"알 수 없는 효과음: {name}")


def load_sfx(name, sfx_dir, work):
    if sfx_dir:
        for ext in (".wav", ".mp3", ".ogg"):
            p = Path(sfx_dir) / f"{name}{ext}"
            if p.exists():
                return to_wav(p, work / f"sfx_{name}.wav")
    return synth_sfx(name).astype(np.float32)


# ---------------------------------------------------------------- 그래픽 (투명 PNG)

def text_png(text, font_path, size, y, out, box=False, color=(255, 255, 255, 255)):
    img = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    words, rows, cur = text.split(), [], ""
    for wd in words:  # 화면 폭 86%를 넘으면 줄바꿈
        trial = f"{cur} {wd}".strip()
        if d.textlength(trial, font=font) > W * 0.86 and cur:
            rows.append(cur)
            cur = wd
        else:
            cur = trial
    rows.append(cur)
    line_h = size * 1.25
    top = y * H - line_h * len(rows) / 2
    for i, row in enumerate(rows):
        tw = d.textlength(row, font=font)
        x, yy = (W - tw) / 2, top + i * line_h
        if box:
            pad = size * 0.3
            d.rounded_rectangle([x - pad, yy - pad * 0.6, x + tw + pad, yy + size + pad * 0.8],
                                radius=pad, fill=RED)
        d.text((x, yy), row, font=font, fill=color,
               stroke_width=0 if box else max(4, size // 9), stroke_fill=(0, 0, 0, 255))
    img.save(out)


def graphic_png(g, font_path, out):
    kind = g["type"]
    if kind in ("label", "badge"):
        size = 96 if kind == "label" else 150
        return text_png(g["text"], font_path, size, g.get("y", 0.3), out, box=(kind == "label"))
    img = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(img)
    px = lambda x, y: (x * W, y * H)
    if kind == "circle":
        cx, cy = px(g["x"], g["y"])
        r = g.get("r", 0.12) * W
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=RED, width=14)
    elif kind == "line":
        d.line([px(g["x1"], g["y1"]), px(g["x2"], g["y2"])], fill=RED, width=g.get("width", 10))
    elif kind == "dimension":  # 양 끝 눈금이 있는 치수선 + 가운데 숫자
        (x1, y1), (x2, y2) = px(g["x1"], g["y1"]), px(g["x2"], g["y2"])
        d.line([(x1, y1), (x2, y2)], fill=RED, width=8)
        ang = math.atan2(y2 - y1, x2 - x1) + math.pi / 2
        tx, ty = math.cos(ang) * 28, math.sin(ang) * 28
        for x, y in ((x1, y1), (x2, y2)):
            d.line([(x - tx, y - ty), (x + tx, y + ty)], fill=RED, width=8)
        if g.get("text"):
            font = ImageFont.truetype(font_path, 80)
            mx, my = (x1 + x2) / 2 + tx * 2.2, (y1 + y2) / 2 + ty * 2.2
            d.text((mx, my), g["text"], font=font, fill=RED, anchor="mm",
                   stroke_width=6, stroke_fill=(255, 255, 255, 255))
    elif kind == "flash":  # 화면 가장자리 빨간 테두리
        d.rectangle([0, 0, W - 1, H - 1], outline=RED, width=40)
    else:
        sys.exit(f"알 수 없는 그래픽: {kind}")
    img.save(out)


# ---------------------------------------------------------------- 조립

def build(ep_path, clip_dir, out, tts_mode, font, sfx_dir, bgm, order="name",
          voicebox_url="http://127.0.0.1:17493", voicebox_profile=None, voice_dir=None):
    ep = json.loads(Path(ep_path).read_text(encoding="utf-8"))
    cfg = {"voice": ep.get("voice", "ko-KR-InJoonNeural"), "rate": ep.get("rate", "+25%"),
           "speed": ep.get("speed", 1.0), "voicebox": {**ep.get("voicebox", {}), "url": voicebox_url}}
    if voicebox_profile:
        cfg["voicebox"]["profile_id"] = voicebox_profile
    if tts_mode == "voicebox" and not cfg["voicebox"].get("profile_id"):
        try:
            profiles = voicebox_profiles(voicebox_url)
        except Exception as e:
            sys.exit(f"Voicebox({voicebox_url})에 연결하지 못했습니다. 앱을 켜 두세요. ({e})")
        print("쓸 목소리를 --voicebox-profile <id> 로 고르거나 episode.json의 voicebox.profile_id에 넣으세요:")
        for pid, name in profiles:
            print(f"  {pid}  {name}")
        sys.exit(1)
    if tts_mode == "files" and not voice_dir:
        sys.exit("--tts files 는 --voice-dir 폴더가 필요합니다 (L000.wav, L001.wav ...).")
    gap = ep.get("line_gap", 0.08)
    font = pick_font(font)
    work = Path(tempfile.mkdtemp(prefix="short_"))
    tts_dir = Path(out).with_suffix("").parent / "_tts_cache"
    tts_dir.mkdir(parents=True, exist_ok=True)
    print(f"작업 폴더: {work}")

    scenes = ep["scenes"]
    clips = match_clips(clip_dir, [s.get("clip", s["id"]) for s in scenes], order)
    all_lines = [ln for s in scenes for ln in s["lines"]]
    voices = narrate(all_lines, cfg, tts_dir, tts_mode, voice_dir)

    # 줄별 시작 시각 계산
    t, li, timeline = 0.0, 0, []
    for s in scenes:
        start, line_starts = t, []
        for _ in s["lines"]:
            line_starts.append(t)
            t += len(voices[li]) / SR + gap
            li += 1
        t += s.get("pad", 0)  # 마지막 대사 뒤 여운 (그래픽이 보일 시간 확보)
        timeline.append((start, t, line_starts))
    total = t
    print(f"완성본 길이: {total:.1f}초")

    # 1) 장면 영상: 세로로 꽉 채워 자르고, 짧으면 느리게 늘린 뒤 길이에 맞춤
    parts = []
    for i, (s, (st, en, _)) in enumerate(zip(scenes, timeline)):
        need, src = en - st, clips[s.get("clip", s["id"])]
        slow = max(1.0, need / max(media_duration(src) - 0.05, 0.1))
        vf = (f"setpts={slow:.4f}*PTS,scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},fps={FPS},tpad=stop_mode=clone:stop_duration=2,trim=duration={need:.3f},"
              f"setsar=1,format=yuv420p")
        part = work / f"part{i:02d}.mp4"
        run(["-i", src, "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", part])
        parts.append(part)
        if slow > 1.5:
            print(f"  {s['id']}: 영상이 짧아 {slow:.1f}배 느리게 늘렸습니다 (검수 때 확인)")
    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    base = work / "base.mp4"
    run(["-f", "concat", "-safe", 0, "-i", concat_list, "-c", "copy", base])

    # 2) 자막과 그래픽 PNG + 표시 구간
    overlays, sfx_events, li = [], [], 0
    for s, (st, en, line_starts) in zip(scenes, timeline):
        for k, text in enumerate(s["lines"]):
            a = line_starts[k]
            b = line_starts[k + 1] if k + 1 < len(s["lines"]) else en
            png = work / f"sub{li:03d}.png"
            text_png(text.replace(".", "").strip(), font, 78, ep.get("subtitle_y", 0.74), png)
            overlays.append((png, a, b))
            li += 1
        for j, g in enumerate(s.get("graphics", [])):
            a = line_starts[g.get("line", 0)] + g.get("offset", 0)
            b = a + g["duration"] if "duration" in g else en
            png = work / f"g_{s['id']}_{j}.png"
            graphic_png(g, font, png)
            overlays.append((png, a, min(b, en)))
        for fx in s.get("sfx", []):
            sfx_events.append((fx["name"], line_starts[fx.get("line", 0)] + fx.get("offset", 0), fx.get("gain", 1.0)))

    # 3) 오디오: 내레이션 + 효과음 + 배경음(선택), numpy로 섞음
    n = int((total + 0.5) * SR)
    voice = np.zeros(n, np.float32)
    li = 0
    for _, _, line_starts in timeline:
        for a in line_starts:
            seg = voices[li]
            i0 = int(a * SR)
            voice[i0:i0 + len(seg)] += seg[: n - i0]
            li += 1
    fx_track = np.zeros(n, np.float32)
    cache = {}
    for name, a, gain in sfx_events:
        if name not in cache:
            cache[name] = load_sfx(name, sfx_dir, work)
        seg, i0 = cache[name] * gain, max(0, int(a * SR))
        fx_track[i0:i0 + len(seg)] += seg[: n - i0]
    mix = voice + fx_track * ep.get("sfx_gain", 0.8)
    if bgm:
        music = to_wav(bgm, work / "bgm.wav")
        music = np.resize(music, n) * ep.get("bgm_gain", 0.12)
        mix += music
    peak = np.abs(mix).max() or 1
    mix = mix / max(peak, 1) * 0.95
    audio = work / "mix.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    # 4) 최종 합성
    inputs, chain, last = ["-i", base, "-i", audio], [], "0:v"
    for k, (png, a, b) in enumerate(overlays):
        inputs += ["-i", png]
        tag = f"v{k}"
        chain.append(f"[{last}][{k + 2}:v]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})'[{tag}]")
        last = tag
    script = work / "filter.txt"
    script.write_text(";".join(chain), encoding="utf-8")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run([*inputs, "-filter_complex_script", script, "-map", f"[{last}]", "-map", "1:a",
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out])
    print(f"완성: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode.json 경로")
    ap.add_argument("--clips", required=True, help="장면 영상 폴더")
    ap.add_argument("--out", help="출력 mp4 (기본: episode.json 옆 output/final.mp4)")
    ap.add_argument("--tts", choices=["edge", "voicebox", "files", "silent"], default="edge",
                    help="edge=마이크로소프트 무료 음성, voicebox=PC의 Voicebox 앱, "
                         "files=직접 만든 음성 파일(--voice-dir), silent=무음 테스트")
    ap.add_argument("--voicebox-url", default="http://127.0.0.1:17493", help="Voicebox 로컬 API 주소")
    ap.add_argument("--voicebox-profile", help="Voicebox 목소리 id (비우면 목록을 보여줌)")
    ap.add_argument("--voice-dir", help="--tts files 일 때 L000.wav, L001.wav ... 가 있는 폴더")
    ap.add_argument("--font", help="자막 글꼴 .ttf (기본: 맑은 고딕 Bold)")
    ap.add_argument("--sfx-dir", help="효과음 폴더 (ting.wav 같은 이름이 있으면 합성음 대신 사용)")
    ap.add_argument("--bgm", help="배경음악 파일")
    ap.add_argument("--order", choices=["name", "mtime"], default="name",
                    help="파일 이름이 S1..이 아닐 때 짝짓는 순서 (mtime=다운로드한 순서)")
    a = ap.parse_args()
    out = a.out or str(Path(a.episode).parent / "output" / "final.mp4")
    build(a.episode, a.clips, out, a.tts, a.font, a.sfx_dir, a.bgm, a.order,
          a.voicebox_url, a.voicebox_profile, a.voice_dir)


if __name__ == "__main__":
    main()

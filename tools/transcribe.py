#!/usr/bin/env python3
"""Narration -> timed transcript (JSON), fully local.

Silero VAD cuts the narration into phrases, Whisper large-v3-turbo (ONNX export from the
sherpa-onnx GitHub release) transcribes each phrase. Tokens are decoded as raw bytes so Korean
syllables that span several byte-level BPE tokens are not lost.

    pip install sherpa-onnx kaldi-native-fbank onnxruntime imageio-ffmpeg
    python tools/transcribe.py audio/narration.mp3 --out data/transcript.json

Models (~560MB) are downloaded once to ~/.cache/collage-models.
"""
import argparse
import base64
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request
import wave

import numpy as np

REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
CACHE = os.environ.get("COLLAGE_MODEL_DIR", os.path.expanduser("~/.cache/collage-models"))


def ensure_models():
    wdir = os.path.join(CACHE, "sherpa-onnx-whisper-turbo")
    vad = os.path.join(CACHE, "silero_vad.onnx")
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(os.path.join(wdir, "turbo-tokens.txt")):
        tb = os.path.join(CACHE, "whisper-turbo.tar.bz2")
        print("[transcribe] downloading whisper-turbo (~560MB) ...", flush=True)
        urllib.request.urlretrieve(REL + "sherpa-onnx-whisper-turbo.tar.bz2", tb)
        with tarfile.open(tb) as tf:
            tf.extractall(CACHE)
        os.remove(tb)
    if not os.path.exists(vad):
        urllib.request.urlretrieve(REL + "silero_vad.onnx", vad)
    return wdir + "/", vad


def load_16k(path):
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = "ffmpeg"
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp],
                   check=True)
    with wave.open(tmp) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    os.remove(tmp)
    return x, 16000


class Whisper:
    def __init__(self, mdir, language="ko", threads=4):
        import kaldi_native_fbank as knf
        import onnxruntime as ort
        self.knf = knf
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.enc = ort.InferenceSession(mdir + "turbo-encoder.int8.onnx", so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(mdir + "turbo-decoder.int8.onnx", so, providers=["CPUExecutionProvider"])
        m = self.enc.get_modelmeta().custom_metadata_map
        self.n_layer, self.n_ctx, self.n_state = int(m["n_text_layer"]), int(m["n_text_ctx"]), int(m["n_text_state"])
        self.n_mels = int(m["n_mels"])
        self.sot, self.eot, self.translate = int(m["sot"]), int(m["eot"]), int(m["translate"])
        self.no_ts, self.no_speech, self.blank = int(m["no_timestamps"]), int(m["no_speech"]), int(m["blank_id"])
        lang2id = dict(zip(m["all_language_codes"].split(","), map(int, m["all_language_tokens"].split(","))))
        self.sot_seq = list(map(int, m["sot_sequence"].split(","))) + [self.no_ts]
        self.sot_seq[1] = lang2id[language]
        self.sot_seq[2] = int(m["transcribe"])
        self.table = {}
        for line in open(mdir + "turbo-tokens.txt", encoding="utf-8"):
            t, i = line.rstrip("\n").rsplit(" ", 1)
            self.table[int(i)] = base64.b64decode(t)

    def _mel(self, samples):
        opts = self.knf.WhisperFeatureOptions()
        opts.dim = self.n_mels
        fb = self.knf.OnlineWhisperFbank(opts)
        fb.accept_waveform(16000, samples.tolist())
        fb.input_finished()
        f = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])
        ls = np.log10(np.clip(f, 1e-10, None))
        ls = np.maximum(ls, ls.max() - 8.0)
        mel = np.pad((ls + 4.0) / 4.0, ((0, 1500), (0, 0)))
        if mel.shape[0] > 3000:
            mel = np.pad(mel[:2950], ((0, 50), (0, 0)))
        return mel.T[None].astype(np.float32)

    def _suppress(self, lg, initial):
        if initial:
            lg[self.eot] = -np.inf
            lg[self.blank] = -np.inf
        for k in (self.no_ts, self.sot, self.no_speech, self.translate):
            lg[k] = -np.inf

    def __call__(self, samples):
        ck, cv = self.enc.run(None, {"mel": self._mel(samples)})
        kc = np.zeros((self.n_layer, 1, self.n_ctx, self.n_state), np.float32)
        vc = np.zeros_like(kc)
        names = [i.name for i in self.dec.get_inputs()]

        def step(tok, kc, vc, off):
            return self.dec.run(None, {names[0]: np.array([tok], np.int64), names[1]: kc, names[2]: vc,
                                       names[3]: ck, names[4]: cv, names[5]: np.array([off], np.int64)})
        lg, kc, vc = step(self.sot_seq, kc, vc, 0)
        off = len(self.sot_seq)
        lg = lg[0, -1].copy()
        self._suppress(lg, True)
        tid, res = int(lg.argmax()), []
        for _ in range(self.n_ctx - off - 1):
            if tid == self.eot:
                break
            res.append(tid)
            lg, kc, vc = step([tid], kc, vc, off)
            off += 1
            lg = lg[0, -1].copy()
            self._suppress(lg, False)
            tid = int(lg.argmax())
            if len(res) > 8 and len(set(res[-8:])) == 1:
                break
        return b"".join(self.table.get(i, b"") for i in res).decode("utf-8", errors="replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--language", default="ko")
    ap.add_argument("--min-silence", type=float, default=0.30)
    a = ap.parse_args()
    import sherpa_onnx
    mdir, vad_path = ensure_models()
    x, sr = load_16k(a.audio)
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_path
    cfg.silero_vad.min_silence_duration = a.min_silence
    cfg.silero_vad.min_speech_duration = 0.20
    cfg.silero_vad.max_speech_duration = 25
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(60, int(len(x) / sr) + 10))
    segs = []
    ws = cfg.silero_vad.window_size
    for i in range(0, len(x), ws):
        vad.accept_waveform(x[i:i + ws])
        while not vad.empty():
            segs.append((vad.front.start, np.array(vad.front.samples)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        segs.append((vad.front.start, np.array(vad.front.samples)))
        vad.pop()
    asr = Whisper(mdir, a.language)
    pad = int(0.15 * sr)
    out = []
    for st, smp in segs:
        lo, hi = max(0, st - pad), min(len(x), st + len(smp) + pad)
        t0, t1 = st / sr, (st + len(smp)) / sr
        txt = asr(x[lo:hi])
        out.append({"start": round(t0, 2), "end": round(t1, 2), "text": txt})
        print(f"[{t0:7.2f} -> {t1:7.2f}] {txt}", flush=True)
    doc = {"source": os.path.basename(a.audio), "duration": round(len(x) / sr, 2), "segments": out}
    path = a.out or os.path.splitext(a.audio)[0] + ".transcript.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print("[transcribe] wrote", path)


if __name__ == "__main__":
    main()

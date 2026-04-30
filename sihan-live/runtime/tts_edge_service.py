"""Edge TTS: map app voice ids (e.g. alipay_lingguang) to Neural voices; per-sentence prosody via multiple synthesize passes."""

from __future__ import annotations

import io
import re
import random

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import edge_tts

app = FastAPI(title="edge-tts-service")

# 支付宝「灵光」无开放 API：用晓伊（偏灵动清亮）作「灵光 / 纹声」近似
VOICE_MAP = {
    "alipay_lingguang": "zh-CN-XiaoyiNeural",
    "lingguang": "zh-CN-XiaoyiNeural",
    "wensheng": "zh-CN-XiaoyiNeural",
    "female_real_cn": "zh-CN-XiaoxiaoNeural",
    "zh-CN-XiaoxiaoNeural": "zh-CN-XiaoxiaoNeural",
    "zh-CN-XiaoyiNeural": "zh-CN-XiaoyiNeural",
    "zh-CN-XiaochenNeural": "zh-CN-XiaochenNeural",
}


def _parse_percent(s: str, default: int = 0) -> int:
    m = re.search(r"([+-]?\d+)", (s or f"{default}").replace(" ", ""))
    return int(m.group(1)) if m else default


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？…\n":
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
        elif ch == "." and len(buf) >= 2 and buf[-2].isascii() and buf[-2].isalnum():
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out if out else [text]


def sentence_prosody_delta(sentence: str) -> tuple[int, int]:
    """Extra (rate_pct, pitch_hz) versus globals."""
    rate_d = 0
    pitch_d = 0
    L = len(sentence)
    if L <= 6:
        rate_d += 8
    elif L >= 50:
        rate_d -= 12
    elif L >= 32:
        rate_d -= 6
    if sentence.rstrip().endswith(("？", "?")):
        pitch_d += 5
        rate_d -= 3
    if sentence.rstrip().endswith(("！", "!")):
        pitch_d += 3
        rate_d += 2
    if re.search(r"[，、；]", sentence):
        rate_d -= 4
    if re.search(r"[嗯嗯唉呀哦噢诶嘿哼…~]", sentence):
        rate_d -= 6
        pitch_d += 1
    if re.search(r"[害羞喜欢想你心疼委屈]", sentence):
        rate_d -= 5
        pitch_d += 2
    return rate_d, pitch_d


def _fmt_rate(pct: int) -> str:
    pct = max(-28, min(28, pct))
    return f"{pct:+d}%"


def _fmt_pitch(hz: int) -> str:
    hz = max(-10, min(14, hz))
    return f"{hz:+d}Hz"


async def _synthesize_chunk(
    text: str,
    voice: str,
    rate_pct: int,
    pitch_hz: int,
    volume_str: str,
) -> bytes:
    comm_kwargs: dict = {
        "text": text,
        "voice": voice,
        "rate": _fmt_rate(rate_pct),
        "pitch": _fmt_pitch(pitch_hz),
    }
    try:
        communicate = edge_tts.Communicate(volume=volume_str, **comm_kwargs)
    except TypeError:
        communicate = edge_tts.Communicate(**comm_kwargs)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


class SpeechReq(BaseModel):
    input: str
    voice: str = "alipay_lingguang"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    dynamic_prosody: bool = True


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "edge-tts",
        "default_voice_key": "alipay_lingguang",
        "mapped_neural": VOICE_MAP.get("alipay_lingguang"),
        "note": "灵光风格映射到 zh-CN-XiaoyiNeural；按句调节语速/停顿更贴近真人",
    }


@app.post("/v1/audio/speech")
async def speech(req: SpeechReq):
    text = (req.input or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "input is empty"})

    voice_key = (req.voice or "alipay_lingguang").strip()
    voice = VOICE_MAP.get(voice_key, voice_key)

    base_r = _parse_percent(req.rate, 0)
    base_p = _parse_percent(req.pitch, 0)

    pieces: list[bytes] = []

    if req.dynamic_prosody and len(text) <= 3000:
        sentences = split_sentences(text)
        if len(sentences) > 24:
            sentences = [text]
        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue
            rd, pd = sentence_prosody_delta(sent)
            r = base_r + rd + random.randint(-2, 2)
            p = base_p + pd + random.randint(-1, 2)
            audio = await _synthesize_chunk(
                sent.strip(), voice, r, p, req.volume
            )
            if audio:
                pieces.append(audio)
    else:
        audio = await _synthesize_chunk(text, voice, base_r, base_p, req.volume)
        pieces = [audio] if audio else []

    if not pieces:
        return JSONResponse(status_code=500, content={"error": "no audio"})

    combined = io.BytesIO()
    for b in pieces:
        combined.write(b)
    combined.seek(0)
    return StreamingResponse(combined, media_type="audio/mpeg")


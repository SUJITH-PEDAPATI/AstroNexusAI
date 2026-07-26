"""
AstroNexus AI — Voice Pipeline

Full voice I/O pipeline:
    Audio → Whisper STT → text query
    Text answer → XTTS TTS → audio response

Run from project root:
    python -m backend.voice.voice_pipeline transcribe <audio_file>
    python -m backend.voice.voice_pipeline speak <"text to say">
    python -m backend.voice.voice_pipeline full <audio_file>
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Combines Whisper STT and XTTS TTS into one interface.
    Used by VoiceAgent in LangGraph (Week 7).
    """

    def __init__(self) -> None:
        from backend.voice.whisper_service import WhisperService
        from backend.voice.tts_service     import TTSService
        self._stt = WhisperService()
        self._tts = TTSService()

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Audio → text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text string
        """
        result = self._stt.transcribe(audio_path)
        return result["text"]

    def speak(
        self,
        text:     str,
        out_path: str | Path = "output/tts_output.wav",
        ref_wav:  str | Path = "output/tts_reference.wav",
    ) -> Path:
        """
        Text → audio WAV.

        Args:
            text:     Text to synthesise
            out_path: Output WAV path
            ref_wav:  Speaker reference WAV

        Returns:
            Path to output WAV
        """
        return self._tts.speak(text, out_path=out_path, ref_wav=ref_wav)

    def full_round_trip(
        self,
        audio_path: str | Path,
        answer:     str,
    ) -> dict:
        """
        Full voice round trip:
            audio → transcribe → [external processing] → speak → audio

        Args:
            audio_path: Input audio (question)
            answer:     Text answer to speak back

        Returns:
            {
                "query":       str,   transcribed question
                "answer":      str,   the answer text
                "audio_out":   str,   path to spoken answer WAV
                "stt_details": dict,  full Whisper result
            }
        """
        stt_result = self._stt.transcribe(audio_path)
        query      = stt_result["text"]
        logger.info(f"[VoicePipeline] Query: {query}")

        out_path = self.speak(answer)
        logger.info(f"[VoicePipeline] Answer audio → {out_path}")

        return {
            "query":       query,
            "answer":      answer,
            "audio_out":   str(out_path),
            "stt_details": stt_result,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = sys.argv[1:]

    if not args:
        print(
            "Usage:\n"
            "  python -m backend.voice.voice_pipeline transcribe <audio>\n"
            "  python -m backend.voice.voice_pipeline speak <text>\n"
            "  python -m backend.voice.voice_pipeline full <audio>"
        )
        sys.exit(1)

    cmd = args[0]

    if cmd == "transcribe":
        if len(args) < 2:
            print("Usage: transcribe <audio_file>")
            sys.exit(1)

        from backend.voice.whisper_service import WhisperService
        result = WhisperService().transcribe(args[1])

        print(f"\n{'='*50}")
        print("WHISPER TRANSCRIPTION")
        print(f"{'='*50}")
        print(f"  Text     : {result['text']}")
        print(f"  Language : {result['language']}")
        print(f"  Duration : {result['duration']}s")
        print(f"  Model    : {result['model']}  device={result['device']}")
        print(f"{'='*50}\n")

    elif cmd == "speak":
        if len(args) < 2:
            print("Usage: speak <text>")
            sys.exit(1)

        text = " ".join(args[1:])
        from backend.voice.tts_service import TTSService
        out  = TTSService().speak(text)

        print(f"\n{'='*50}")
        print("TTS OUTPUT")
        print(f"{'='*50}")
        print(f"  Text    : {text[:80]}...")
        print(f"  Output  : {out}")
        print(f"{'='*50}\n")

    elif cmd == "full":
        if len(args) < 2:
            print("Usage: full <audio_file>")
            sys.exit(1)

        vp     = VoicePipeline()
        query  = vp.transcribe(args[1])
        answer = f"I received your query: {query}. Processing complete."
        result = vp.full_round_trip(args[1], answer)

        print(f"\n{'='*50}")
        print("VOICE ROUND TRIP")
        print(f"{'='*50}")
        print(f"  Query     : {result['query']}")
        print(f"  Answer    : {result['answer'][:80]}...")
        print(f"  Audio out : {result['audio_out']}")
        print(f"{'='*50}\n")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
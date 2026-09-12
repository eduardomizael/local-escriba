# -*- coding: utf-8 -*-
"""
Transcreve audio e video com faster-whisper, usando a GPU quando disponivel.

    uv sync
    uv run transcribe.py                          # varre a pasta atual
    uv run transcribe.py video.mp4 --output output
    uv run transcribe.py --input D:/videos --model medium
    uv run transcribe.py entrevista.mp3 --start 660 --end 930

Gera .json (segmentos + tempos), .txt e .srt. Por padrao ao lado de cada
arquivo de midia; com --output, na pasta indicada.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Extensoes varridas quando nenhum arquivo e passado na linha de comando.
MEDIA_EXTENSIONS = (
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
)

# Ordem de degradacao: GPU rapida -> GPU economica (metade da VRAM) -> CPU.
DEVICE_FALLBACKS = (("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8"))

# Intervalo, em segundos de audio transcrito, entre as linhas de progresso.
PROGRESS_INTERVAL_S = 300


def setup_cuda() -> None:
    """No Windows, expoe as DLLs de cuBLAS/cuDNN vindas dos wheels da NVIDIA.

    Sem isto o CTranslate2 nao encontra CUDA e cai para CPU silenciosamente,
    mesmo com a GPU presente. Evita instalar o CUDA Toolkit na maquina.

    'nvidia' e um namespace package (PEP 420): nao tem __init__.py, entao
    __file__ e None e so __path__ existe. Nunca levanta excecao — se algo
    falhar aqui, o pior caso e rodar em CPU, e o carregador avisa.
    """
    if sys.platform != "win32":
        return
    try:
        roots: list[Path] = []
        try:
            import nvidia
            roots += [Path(x) for x in getattr(nvidia, "__path__", [])]
        except ImportError:
            pass
        import sysconfig
        paths = sysconfig.get_paths()
        for key in ("purelib", "platlib"):
            if paths.get(key):
                roots.append(Path(paths[key]) / "nvidia")

        folders: set[Path] = set()
        for root in roots:
            if root.is_dir():
                for dll in root.rglob("*.dll"):
                    folders.add(dll.parent)

        for folder in sorted(folders):
            os.add_dll_directory(str(folder))
            os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"

        if folders:
            names = sorted({folder.parent.name for folder in folders})
            print(f"[cuda] {len(folders)} pasta(s) de DLL registrada(s): {', '.join(names)}")
        else:
            print("[cuda] nenhuma DLL da NVIDIA encontrada — deve cair para CPU")
    except Exception as e:
        print(f"[cuda] preparacao ignorada ({type(e).__name__}: {e})")


setup_cuda()

def load_model(name: str):
    """Carrega o modelo na primeira configuracao de dispositivo que funcionar."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Dependencias ausentes. Rode 'uv sync' nesta pasta primeiro.")

    for device, compute_type in DEVICE_FALLBACKS:
        try:
            model = WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"  [--] {device}/{compute_type}: {str(e).splitlines()[0][:110]}")
            continue
        print(f"  [ok] modelo '{name}' carregado em {device}/{compute_type}")
        if device == "cpu":
            print("  [!] rodando em CPU — vai demorar muito mais que na GPU")
        return model
    sys.exit("Nenhuma configuracao de dispositivo funcionou.")


def timestamp(seconds: float, sep: str = ",") -> str:
    """Formata segundos como HH:MM:SS<sep>mmm (',' para o SRT, '.' para o .txt)."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{int((seconds % 1) * 1000):03d}"


def find_media(folder: str) -> list[Path]:
    """Lista, em ordem alfabetica, as midias reconhecidas dentro da pasta."""
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def parse_time(value: str) -> float:
    """Aceita segundos ou um horário no formato HH:MM:SS[.mmm]."""
    if ":" not in value:
        try:
            return float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "must be seconds or HH:MM:SS"
            ) from error

    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("must be seconds or HH:MM:SS")

    hours, minutes, seconds = parts
    try:
        if not hours.isdigit() or not minutes.isdigit():
            raise ValueError
        hours_value = int(hours)
        minutes_value = int(minutes)
        seconds_value = float(seconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be seconds or HH:MM:SS"
        ) from error

    if minutes_value >= 60 or not 0 <= seconds_value < 60:
        raise argparse.ArgumentTypeError("minutes and seconds must be below 60")
    return hours_value * 3600 + minutes_value * 60 + seconds_value


def clip_timestamps(start: float | None, end: float | None) -> str | None:
    """Converte limites opcionais para o formato aceito pelo faster-whisper."""
    if start is None and end is None:
        return None

    values = [0.0 if start is None else start]
    if end is not None:
        values.append(end)
    return ",".join(f"{value:g}" for value in values)


def range_suffix(start: float | None, end: float | None) -> str:
    """Cria um sufixo estável para evitar colisões entre transcrições parciais."""
    parts = []
    if start is not None:
        parts.append(f"start-{start:g}")
    if end is not None:
        parts.append(f"end-{end:g}")
    return "" if not parts else "__" + "__".join(parts)


def validate_range(parser: argparse.ArgumentParser,
                   start: float | None, end: float | None) -> None:
    """Valida limites que independem da duração da mídia."""
    if start is not None and not math.isfinite(start):
        parser.error("--start must be a finite number")
    if end is not None and not math.isfinite(end):
        parser.error("--end must be a finite number")
    if start is not None and start < 0:
        parser.error("--start must be greater than or equal to zero")
    if end is not None and end < 0:
        parser.error("--end must be greater than or equal to zero")
    if start is not None and end is not None and end <= start:
        parser.error("--end must be greater than --start")


def decode_audio_range(media: Path, start: float | None, end: float | None,
                       sampling_rate: int = 16000):
    """Decodifica somente o intervalo solicitado, preservando a duração original.

    O ``clip_timestamps`` do faster-whisper limita a inferência, mas a versão
    usada pelo projeto ainda calcula o espectrograma completo antes do corte.
    Aqui o PyAV busca e decodifica apenas a faixa solicitada, evitando esse
    consumo de memória em gravações longas.
    """
    try:
        import av
        import numpy as np
    except ImportError:
        sys.exit("Dependencias ausentes. Rode 'uv sync' nesta pasta primeiro.")

    with av.open(str(media), mode="r", metadata_errors="ignore") as container:
        try:
            stream = container.streams.audio[0]
        except IndexError as error:
            raise ValueError("No audio stream found in the media") from error

        if container.duration is not None:
            source_duration = float(container.duration / av.time_base)
        elif stream.duration is not None:
            source_duration = float(stream.duration * stream.time_base)
        else:
            raise ValueError("Could not determine the media duration")

        clip_start = 0.0 if start is None else start
        clip_end = source_duration if end is None else end
        if clip_start >= source_duration:
            raise ValueError(f"--start ({clip_start:g}) must be before the end of the media "
                             f"({source_duration:.2f}s)")
        if clip_end > source_duration:
            raise ValueError(f"--end ({clip_end:g}) exceeds the media duration "
                             f"({source_duration:.2f}s)")

        if clip_start > 0:
            container.seek(int(clip_start * av.time_base), backward=True)

        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=sampling_rate,
        )
        chunks = []
        reached_end = False
        for decoded in container.decode(stream):
            for frame in resampler.resample(decoded):
                if frame.time is None:
                    continue
                frame_start = float(frame.time)
                frame_end = frame_start + frame.samples / frame.sample_rate
                if frame_end <= clip_start:
                    continue
                if frame_start >= clip_end:
                    reached_end = True
                    break
                first = max(0, int(round((clip_start - frame_start) * frame.sample_rate)))
                last = min(frame.samples, int(round((clip_end - frame_start) * frame.sample_rate)))
                if last > first:
                    chunks.append(frame.to_ndarray()[..., first:last].reshape(-1).copy())
            if reached_end:
                break

    if not chunks:
        raise ValueError("No audio samples found in the requested range")
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    return audio, source_duration, clip_start, clip_end


def transcribe_file(model, media: Path, model_name: str,
                    output_dir: Path | None, language: str | None,
                    start: float | None = None, end: float | None = None) -> None:
    """Transcreve um arquivo e grava .json, .txt e .srt. Pula se o .json existir."""
    destination = output_dir if output_dir is not None else media.parent
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / f"{media.stem}{range_suffix(start, end)}"
    json_target = base.with_suffix(".json")
    if output_dir is not None:
        print(f"  saida em: {destination}")
    if json_target.exists():
        print(f"[skip] {json_target.name} ja existe")
        return

    print(f"\n=== {media.name}")
    started = time.time()
    timestamps = clip_timestamps(start, end)
    source_duration = None
    range_start = 0.0 if start is None else start
    range_end = None
    audio = str(media)
    if timestamps is not None:
        audio, source_duration, range_start, range_end = decode_audio_range(
            media, start, end,
        )
    transcribe_options = dict(
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    segments, info = model.transcribe(
        audio,
        **transcribe_options,
    )
    if source_duration is None:
        source_duration = info.duration
    if range_end is None:
        range_end = source_duration
    range_duration = range_end - range_start
    if timestamps is None:
        print(f"  duracao da midia: {source_duration / 60:.0f} min"
              f"  |  idioma: {info.language}")
    else:
        print(f"  duracao da midia: {source_duration / 60:.0f} min"
              f"  |  intervalo: {timestamp(range_start, '.')[:8]} -> "
              f"{timestamp(range_end, '.')[:8]} ({range_duration / 60:.1f} min)"
              f"  |  idioma: {info.language}")

    rows: list[dict] = []
    last_report = 0.0
    # Escreve em .parcial e renomeia no fim: uma interrupcao nao deixa
    # arquivo truncado se passando por completo.
    tmp_txt = base.with_suffix(".txt.parcial")
    tmp_srt = base.with_suffix(".srt.parcial")
    with open(tmp_txt, "w", encoding="utf-8") as f_txt, \
        open(tmp_srt, "w", encoding="utf-8") as f_srt:
        for index, segment in enumerate(segments, 1):
            text = segment.text.strip()
            segment_start = segment.start + range_start
            segment_end = segment.end + range_start
            rows.append({"i": index, "start": round(segment_start, 2),
                         "end": round(segment_end, 2), "text": text})
            f_txt.write(f"[{timestamp(segment_start, '.')[:8]}] {text}\n")
            f_srt.write(f"{index}\n{timestamp(segment_start)} --> "
                        f"{timestamp(segment_end)}\n{text}\n\n")
            processed = max(segment_end - range_start, 0.0)
            if processed - last_report >= PROGRESS_INTERVAL_S:
                last_report = processed
                elapsed = max(time.time() - started, 1e-6)
                speed = max(processed / elapsed, 1e-6)
                pct = 100 * processed / range_duration if range_duration else 0
                remaining = max(range_end - segment_end, 0) / speed / 60
                print(f"  {pct:5.1f}%  |  {processed / 60:4.1f} / "
                      f"{range_duration / 60:.1f} min do intervalo"
                      f"  |  {speed:5.1f}x tempo real"
                      f"  |  faltam ~{remaining:.0f} min")

    tmp_txt.replace(base.with_suffix(".txt"))
    tmp_srt.replace(base.with_suffix(".srt"))
    json_target.write_text(
        json.dumps({"file": media.name,
                    "duration_s": round(source_duration, 1),
                    "model": model_name,
                    "language": info.language,
                    "requested_range_s": (
                        None if timestamps is None else {"start": start, "end": end}
                    ),
                    "segments": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    scope = "da mídia completa" if timestamps is None else f"do intervalo de {range_duration / 60:.1f} min"
    print(f"[pronto] {len(rows)} segmentos {scope} em "
          f"{(time.time() - started) / 60:.1f} min")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcreve audio e video com faster-whisper.")
    parser.add_argument("files", nargs="*",
                        help="arquivos a transcrever (padrao: varre --input)")
    parser.add_argument("--model", default="large-v3",
                        help="large-v3 (padrao), medium, small, base, tiny...")
    parser.add_argument("--input", default=".",
                        help="pasta onde procurar midias se nenhum arquivo for passado")
    parser.add_argument("--output", default=None,
                        help="pasta onde gravar .json/.txt/.srt (padrao: junto da midia)")
    parser.add_argument("--language", default="pt",
                        help="codigo do idioma, ex.: pt, en, es. Use 'auto' para detectar")
    parser.add_argument("--start", type=parse_time, default=None,
                        help="start time in seconds or HH:MM:SS (default: beginning)")
    parser.add_argument("--end", type=parse_time, default=None,
                        help="end time in seconds or HH:MM:SS (default: end)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_range(parser, args.start, args.end)

    if args.files:
        files = [Path(f).resolve() for f in args.files]
    else:
        files = find_media(args.input)

    missing = [f for f in files if not f.exists()]
    if missing:
        sys.exit("Nao encontrei: " + ", ".join(str(f) for f in missing))
    if not files:
        sys.exit(f"Nenhuma midia reconhecida em {Path(args.input).resolve()}")

    print("Arquivos a transcrever:")
    for f in files:
        print(f"  - {f.name}  ({f.stat().st_size / 2**30:.2f} GB)")

    output_dir = Path(args.output).resolve() if args.output else None
    language = None if args.language.lower() == "auto" else args.language
    model = load_model(args.model)
    try:
        for f in files:
            transcribe_file(model, f, args.model, output_dir, language, args.start, args.end)
    except ValueError as error:
        sys.exit(str(error))
    print("\nTudo pronto.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Transcreve audio e video com faster-whisper, usando a GPU quando disponivel.

    uv sync
    uv run transcribe.py                          # varre a pasta atual
    uv run transcribe.py video.mp4 --output output
    uv run transcribe.py --input D:/videos --model medium

Gera .json (segmentos + tempos), .txt e .srt. Por padrao ao lado de cada
arquivo de midia; com --output, na pasta indicada.
"""
from __future__ import annotations

import argparse
import json
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

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("Dependencias ausentes. Rode 'uv sync' nesta pasta primeiro.")


def load_model(name: str):
    """Carrega o modelo na primeira configuracao de dispositivo que funcionar."""
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


def transcribe_file(model, media: Path, model_name: str,
                    output_dir: Path | None, language: str | None) -> None:
    """Transcreve um arquivo e grava .json, .txt e .srt. Pula se o .json existir."""
    destination = output_dir if output_dir is not None else media.parent
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / media.stem
    json_target = base.with_suffix(".json")
    if output_dir is not None:
        print(f"  saida em: {destination}")
    if json_target.exists():
        print(f"[skip] {json_target.name} ja existe")
        return

    print(f"\n=== {media.name}")
    started = time.time()
    segments, info = model.transcribe(
        str(media),
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    print(f"  duracao: {info.duration / 60:.0f} min  |  idioma: {info.language}")

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
            rows.append({"i": index, "start": round(segment.start, 2),
                         "end": round(segment.end, 2), "text": text})
            f_txt.write(f"[{timestamp(segment.start, '.')[:8]}] {text}\n")
            f_srt.write(f"{index}\n{timestamp(segment.start)} --> "
                        f"{timestamp(segment.end)}\n{text}\n\n")
            if segment.end - last_report >= PROGRESS_INTERVAL_S:
                last_report = segment.end
                elapsed = max(time.time() - started, 1e-6)
                speed = max(segment.end / elapsed, 1e-6)
                pct = 100 * segment.end / info.duration if info.duration else 0
                remaining = max(info.duration - segment.end, 0) / speed / 60
                print(f"  {pct:5.1f}%  |  {segment.end / 60:5.0f} min de audio"
                      f"  |  {speed:5.1f}x tempo real"
                      f"  |  faltam ~{remaining:.0f} min")

    tmp_txt.replace(base.with_suffix(".txt"))
    tmp_srt.replace(base.with_suffix(".srt"))
    json_target.write_text(
        json.dumps({"file": media.name,
                    "duration_s": round(info.duration, 1),
                    "model": model_name,
                    "language": info.language,
                    "segments": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[pronto] {len(rows)} segmentos em {(time.time() - started) / 60:.1f} min")


def main() -> None:
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
    args = parser.parse_args()

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
    for f in files:
        transcribe_file(model, f, args.model, output_dir, language)
    print("\nTudo pronto.")


if __name__ == "__main__":
    main()

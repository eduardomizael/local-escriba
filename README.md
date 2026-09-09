# local-escriba

Transcreve áudio e vídeo na sua própria máquina, com
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) e aceleração por GPU.
Sem mensalidade, sem conta em serviço nenhum e sem subir o áudio para lugar
algum — o que costuma decidir a questão quando a gravação é confidencial.

Para cada arquivo gera três saídas: `.json`, `.txt` e `.srt`. Usa a GPU quando
ela está disponível e cai para a CPU quando não está. Feito para o caso chato:
gravações longas, de horas, que precisam virar texto.

## Requisitos

- Python 3.10 a 3.12
- [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente
- Opcional, mas é o ponto: uma GPU NVIDIA com cerca de 5 GB de VRAM livres para
  o modelo `large-v3`. Sem GPU o script funciona, só que bem mais devagar.

Não é preciso instalar o CUDA Toolkit nem o ffmpeg: as bibliotecas de CUDA vêm
como wheels do pip, e a decodificação de áudio é feita pelo PyAV, que já
acompanha o faster-whisper.

## Instalação

```bash
git clone https://github.com/<seu-usuario>/local-escriba.git
cd local-escriba
uv sync
```

Na primeira execução o modelo é baixado do Hugging Face (~3 GB no caso do
`large-v3`) e fica em cache em `~/.cache/huggingface`.

## Uso

Varrer a pasta atual e transcrever tudo que encontrar:

```bash
uv run transcribe.py
```

Arquivos específicos, com as saídas em outra pasta:

```bash
uv run transcribe.py palestra.mp4 entrevista.mp3 --output output
```

Outra pasta de entrada, com um modelo menor:

```bash
uv run transcribe.py --input D:/videos --model medium
```

## Opções

| Opção | Padrão | O que faz |
|---|---|---|
| `--model` | `large-v3` | tamanho do modelo: `large-v3`, `medium`, `small`, `base`, `tiny` |
| `--input` | `.` | pasta varrida quando nenhum arquivo é passado |
| `--output` | ao lado da mídia | pasta onde gravar os resultados |
| `--language` | `pt` | código do idioma; `auto` deixa o modelo detectar |

Se faltar VRAM, `--model medium` costuma resolver.

Extensões reconhecidas na varredura: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`,
`.m4v`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`. Passando o
caminho direto, qualquer formato que o PyAV abra funciona.

## O que sai

Três arquivos por mídia:

| Arquivo | Conteúdo |
|---|---|
| `.json` | segmentos com início e fim em segundos, para processar depois |
| `.txt` | texto corrido com marcação de tempo a cada fala |
| `.srt` | legenda, para reassistir com o texto |

O `.json` tem esta forma:

```json
{
 "file": "palestra.mp4",
 "duration_s": 5412.3,
 "model": "large-v3",
 "language": "pt",
 "segments": [
  { "i": 1, "start": 0.0, "end": 4.2, "text": "Bom dia a todos." }
 ]
}
```

## Detalhes de implementação

**CUDA sem instalar o CUDA Toolkit.** As dependências incluem os wheels
`nvidia-cublas-cu12` e `nvidia-cudnn-cu12`. No Windows o CTranslate2 não acha
essas DLLs sozinho, então `setup_cuda()` registra os diretórios delas via
`os.add_dll_directory` antes de importar o `faster_whisper`. Sem isso o modelo
carrega, mas silenciosamente em CPU. No Linux o equivalente é apontar o
`LD_LIBRARY_PATH` para os mesmos diretórios antes de rodar:

```bash
export LD_LIBRARY_PATH=$(uv run python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

**Degradação em cascata.** Tenta `cuda/float16`, depois `cuda/int8_float16`
(metade da VRAM), depois `cpu/int8`. Avisa em voz alta se cair para CPU.

**Retomável.** Pula qualquer mídia que já tenha `.json` na pasta de saída. Os
`.txt` e `.srt` são escritos como `.parcial` e renomeados só no fim, então uma
interrupção não deixa arquivo truncado se passando por completo.

**Progresso.** A cada 5 minutos de áudio transcrito, imprime percentual,
velocidade em múltiplos do tempo real e estimativa do que falta.

**Parâmetros deliberados.** `condition_on_previous_text=False` evita que o
modelo entre em loop repetindo frases em trechos de silêncio ou palmas, comum
em gravação de evento; `vad_filter=True` pula os silêncios.

## Licença

MIT — veja [LICENSE](LICENSE).

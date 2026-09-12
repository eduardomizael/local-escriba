import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout


MODULE_PATH = Path(__file__).parents[1] / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("local_escriba", MODULE_PATH)
transcribe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transcribe)


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, media, **options):
        self.calls.append((media, options))
        segments = [SimpleNamespace(start=12.5, end=18.0, text=" Trecho de teste. ")]
        return segments, SimpleNamespace(duration=100.0, language="pt")


class RangeOptionsTests(unittest.TestCase):
    def test_parser_accepts_seconds_and_hhmmss(self):
        args = transcribe.build_parser().parse_args([
            "--start", "00:11:00", "--end", "01:02:03.5",
        ])

        self.assertEqual(args.start, 660.0)
        self.assertEqual(args.end, 3723.5)
        self.assertEqual(transcribe.parse_time("660.5"), 660.5)

    def test_parser_rejects_invalid_clock_values(self):
        parser = transcribe.build_parser()
        for value in ("11:00", "00:60:00", "00:00:60", "aa:00:00"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--start", value])

    def test_clip_timestamps_supports_open_and_closed_ranges(self):
        self.assertIsNone(transcribe.clip_timestamps(None, None))
        self.assertEqual(transcribe.clip_timestamps(12.5, None), "12.5")
        self.assertEqual(transcribe.clip_timestamps(None, 30.0), "0,30")
        self.assertEqual(transcribe.clip_timestamps(12.5, 30.0), "12.5,30")

    def test_range_suffix_distinguishes_partial_outputs(self):
        self.assertEqual(transcribe.range_suffix(None, None), "")
        self.assertEqual(transcribe.range_suffix(12.5, None), "__start-12.5")
        self.assertEqual(transcribe.range_suffix(None, 30.0), "__end-30")
        self.assertEqual(transcribe.range_suffix(12.5, 30.0), "__start-12.5__end-30")

    def test_invalid_ranges_are_rejected(self):
        parser = transcribe.build_parser()
        for start, end in (
            (-1.0, None), (None, -1.0), (10.0, 10.0), (20.0, 10.0),
            (float("nan"), None), (None, float("inf")),
        ):
            with self.assertRaises(SystemExit):
                transcribe.validate_range(parser, start, end)


class PartialTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.media = self.folder / "entrevista.mp3"
        self.media.touch()

    def test_full_transcription_keeps_current_vad_and_file_name(self):
        model = FakeModel()

        transcribe.transcribe_file(model, self.media, "small", self.folder, "pt")

        _, options = model.calls[0]
        self.assertTrue(options["vad_filter"])
        self.assertNotIn("clip_timestamps", options)
        payload = json.loads((self.folder / "entrevista.json").read_text(encoding="utf-8"))
        self.assertIsNone(payload["requested_range_s"])

    def test_closed_range_is_sent_to_model_and_recorded(self):
        model = FakeModel()

        output = io.StringIO()
        with redirect_stdout(output), patch.object(
            transcribe, "decode_audio_range", return_value=("selected audio", 100.0, 10.0, 30.0),
        ) as decode_range:
            transcribe.transcribe_file(
                model, self.media, "small", self.folder, "pt", start=10.0, end=30.0
            )

        audio, options = model.calls[0]
        self.assertEqual(audio, "selected audio")
        self.assertTrue(options["vad_filter"])
        self.assertNotIn("clip_timestamps", options)
        decode_range.assert_called_once_with(self.media, 10.0, 30.0)
        target = self.folder / "entrevista__start-10__end-30.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["requested_range_s"], {"start": 10.0, "end": 30.0})
        self.assertEqual(payload["segments"][0]["start"], 22.5)
        self.assertIn("intervalo: 00:00:10 -> 00:00:30 (0.3 min)", output.getvalue())

    def test_open_ranges_are_sent_in_faster_whisper_format(self):
        start_model = FakeModel()
        end_model = FakeModel()

        with patch.object(
            transcribe, "decode_audio_range", return_value=("from start", 100.0, 10.0, 100.0),
        ) as start_decode:
            transcribe.transcribe_file(
                start_model, self.media, "small", self.folder, "pt", start=10.0
            )
        with patch.object(
            transcribe, "decode_audio_range", return_value=("until end", 100.0, 0.0, 30.0),
        ) as end_decode:
            transcribe.transcribe_file(
                end_model, self.media, "small", self.folder, "pt", end=30.0
            )

        self.assertEqual(start_model.calls[0][0], "from start")
        self.assertEqual(end_model.calls[0][0], "until end")
        start_decode.assert_called_once_with(self.media, 10.0, None)
        end_decode.assert_called_once_with(self.media, None, 30.0)

    def test_duration_bounds_are_rejected_before_outputs_are_written(self):
        model = FakeModel()

        with patch.object(transcribe, "decode_audio_range", side_effect=ValueError("--end")):
            with self.assertRaisesRegex(ValueError, "--end"):
                transcribe.transcribe_file(
                    model, self.media, "small", self.folder, "pt", end=101.0
                )

        self.assertFalse((self.folder / "entrevista__end-101.json").exists())


if __name__ == "__main__":
    unittest.main()

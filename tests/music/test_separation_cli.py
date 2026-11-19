"""Check that the optional separation stage passes its WAV to transcription."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from click.testing import CliRunner
from omnizart.cli.music.transcribe import transcribe


class SeparationCliTests(unittest.TestCase):
    def test_vocal_track_can_use_pymss_without_reducing_instrument_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            def isolate(source, base, model, **kwargs):
                self.assertEqual(Path(source), audio)
                output = Path(str(base) + '_vocals.wav')
                output.touch()
                return [output]
            with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate) as separator, \
                 patch('omnizart.cli.music.transcribe._transcribe') as instrument, \
                 patch('omnizart.desktop.combined.add_vocals') as combine:
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Piano',
                    '--output', str(root / 'result'), '--include-vocals',
                    '--vocal-track-separator', 'pymss', '--vocal-track-pymss-model', 'voice'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(Path(instrument.call_args.args[0]), audio)
            self.assertEqual(Path(combine.call_args.args[1]).name, 'pymss_vocals.wav')
            separator.assert_called_once()

    def test_independent_pymss_models_use_original_audio_for_reduction_and_vocal_track(self):
        import numpy as np
        import soundfile as sf
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            seen = []
            def isolate(source, base, model, **kwargs):
                seen.append((Path(source), model))
                outputs = [Path(str(base) + '_' + name + '.wav') for name in ('vocals', 'other')]
                for path, value in zip(outputs, (.1 if model == 'reduction' else .25, .5)):
                    sf.write(str(path), np.full((20, 2), value, dtype='float32'), 44100, subtype='FLOAT')
                return outputs
            def instrument(source, _model, target, *args):
                data, _ = sf.read(str(source), dtype='float32')
                np.testing.assert_allclose(data, .5, atol=1e-6)
            def combine(source, vocal, instrument_path, output):
                self.assertEqual(Path(source), audio)
                data, _ = sf.read(str(vocal), dtype='float32')
                np.testing.assert_allclose(data, .25, atol=1e-6)
            with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate), \
                 patch('omnizart.cli.music.transcribe._transcribe', side_effect=instrument), \
                 patch('omnizart.desktop.combined.add_vocals', side_effect=combine):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Piano',
                    '--output', str(root / 'result'), '--remove-vocals', '--include-vocals',
                    '--separator', 'pymss', '--reduction-pymss-model', 'reduction',
                    '--vocal-track-separator', 'pymss', '--vocal-track-pymss-model', 'voice'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(seen, [(audio, 'reduction'), (audio, 'voice')])

    def test_piano_stem_and_vocal_track_reuse_one_pymss_pass(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            def isolate(source, base, model, **kwargs):
                self.assertEqual(Path(source), audio)
                self.assertIsNone(kwargs['selected_stems'])
                outputs = [Path(str(base) + '_' + name + '.wav') for name in ('piano', 'vocals')]
                for path in outputs:
                    path.write_bytes(path.stem.encode())
                return outputs
            def instrument(source, _model, target, *args):
                self.assertEqual(Path(source).name, 'separated_piano.wav')
            def combine(source, vocal, instrument_path, output):
                self.assertEqual(Path(source), audio)
                self.assertEqual(Path(vocal).name, 'separated_vocals.wav')
            with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate) as separator, \
                 patch('omnizart.cli.music.transcribe._transcribe', side_effect=instrument), \
                 patch('omnizart.desktop.combined.add_vocals', side_effect=combine):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Piano',
                    '--output', str(root / 'result'), '--piano-only', '--piano-separator', 'pymss',
                    '--pymss-model', 'shared', '--include-vocals', '--vocal-track-separator', 'pymss',
                    '--vocal-track-pymss-model', 'shared'])
            self.assertEqual(result.exit_code, 0, result.output)
            separator.assert_called_once()

    def test_six_stem_piano_and_vocals_share_one_pass(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            def isolate(source, base, **kwargs):
                self.assertEqual(Path(source), audio)
                self.assertEqual(kwargs['selected_stems'], ['piano', 'vocals'])
                for name in ('piano', 'vocals'):
                    Path(str(base) + '_' + name + '.wav').touch()
            with patch('omnizart.desktop.six_stems.separate', side_effect=isolate) as separator, \
                 patch('omnizart.cli.music.transcribe._transcribe'), \
                 patch('omnizart.desktop.combined.add_vocals') as combine:
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Piano',
                    '--output', str(root / 'result'), '--piano-only', '--include-vocals',
                    '--vocal-track-separator', 'stem'])
            self.assertEqual(result.exit_code, 0, result.output)
            separator.assert_called_once()
            self.assertEqual(Path(combine.call_args.args[1]).name, 'separated_vocals.wav')

    def test_piano_and_voice_are_combined_in_midi_and_musicxml(self):
        import mido
        import pretty_midi
        from xml.etree import ElementTree as ET
        from omnizart.desktop.combined import merge_tracks
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            def isolate(source, base, model, **kwargs):
                paths = [Path(str(base) + '_' + name + '.wav') for name in ('piano', 'vocals')]
                for path in paths:
                    path.touch()
                return paths
            def instrument(source, _model, target, *args):
                midi = mido.MidiFile()
                midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80),
                                                   mido.Message('note_off', note=60, time=480)]))
                midi.save(str(target))
            def combine(source, vocal, instrument_path, output):
                voice = pretty_midi.PrettyMIDI()
                vocal_track = pretty_midi.Instrument(program=0)
                vocal_track.notes.append(pretty_midi.Note(80, 72, 0, .5))
                voice.instruments.append(vocal_track)
                merge_tracks(instrument_path, voice, str(output) + '.mid')
            with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate), \
                 patch('omnizart.cli.music.transcribe._transcribe', side_effect=instrument), \
                 patch('omnizart.desktop.combined.add_vocals', side_effect=combine):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Piano',
                    '--output', str(root / 'result'), '--piano-only', '--piano-separator', 'pymss',
                    '--pymss-model', 'shared', '--include-vocals', '--vocal-track-separator', 'stem',
                    '--score', '--score-key', 'C'])
            self.assertEqual(result.exit_code, 0, result.output)
            midi = pretty_midi.PrettyMIDI(str(root / 'result.mid'))
            self.assertEqual(sorted(n.pitch for track in midi.instruments for n in track.notes), [60, 72])
            score = ET.parse(root / 'result.musicxml')
            self.assertEqual(len(score.findall('.//part')), 2)
            self.assertIn('Vocals', [x.text for x in score.findall('.//part-name')])


    def test_pymss_vocals_and_accompaniment_feed_combined_midi(self):
        import mido
        import numpy as np
        import pretty_midi
        import soundfile as sf
        from omnizart.desktop.combined import merge_tracks
        for device in ('cpu', 'cuda'):
            with self.subTest(device=device), TemporaryDirectory() as directory:
                root = Path(directory)
                audio = root / 'song.wav'
                audio.touch()
                def isolate(source, base, model, **kwargs):
                    self.assertEqual((Path(source), model), (audio, 'vocal-model'))
                    if device == 'cuda':
                        self.assertEqual((kwargs['device'], kwargs['gpu_index']), ('cuda', 1))
                    outputs = [Path(str(base) + '_' + name + '.wav') for name in ('vocals', 'piano', 'drums')]
                    for path, value in zip(outputs, (.1, .3, .2)):
                        sf.write(str(path), np.full((20, 2), value, dtype='float32'), 44100, subtype='FLOAT')
                    return outputs
                def instrument(source, _model, target, *args):
                    data, _ = sf.read(str(source), dtype='float32')
                    np.testing.assert_allclose(data, .5, atol=1e-6)
                    midi = mido.MidiFile()
                    midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80),
                                                       mido.Message('note_off', note=60, time=480)]))
                    midi.save(str(target))
                def combine(_audio, vocal_path, instrument_path, output):
                    vocals, _ = sf.read(str(vocal_path), dtype='float32')
                    np.testing.assert_allclose(vocals, .1, atol=1e-6)
                    voice = pretty_midi.PrettyMIDI()
                    vocal_track = pretty_midi.Instrument(program=0)
                    vocal_track.notes.append(pretty_midi.Note(80, 72, 0, .5))
                    voice.instruments.append(vocal_track)
                    merge_tracks(instrument_path, voice, str(output) + '.mid')
                args = [str(audio), '--model-path', 'Piano', '--output', str(root / 'result'),
                        '--remove-vocals', '--include-vocals', '--separator', 'pymss',
                        '--pymss-model', 'vocal-model', '--save-accompaniment']
                if device == 'cuda':
                    args += ['--pymss-device', 'cuda', '--gpu-index', '1']
                with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate), \
                     patch('omnizart.cli.music.transcribe._transcribe', side_effect=instrument), \
                     patch('omnizart.desktop.combined.add_vocals', side_effect=combine):
                    result = CliRunner().invoke(transcribe, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue((root / 'result.mid').is_file())
                self.assertTrue((root / 'result_accompaniment.wav').is_file())
                tracks = pretty_midi.PrettyMIDI(str(root / 'result.mid')).instruments
                self.assertEqual([n.pitch for track in tracks for n in track.notes], [60, 72])
                self.assertEqual(tracks[-1].name, 'Vocals')
                self.assertFalse((root / 'result_vocals.wav').exists())

    def test_selected_stem_transcribed_and_only_saved_when_requested(self):
        import mido
        for keep in (False, True):
            with self.subTest(keep=keep), TemporaryDirectory() as directory:
                root = Path(directory)
                audio = root / 'song.wav'
                audio.touch()
                seen = []
                def isolate(_source, base, _model, selected_stems, source_name):
                    self.assertEqual(selected_stems, ['vocals'])
                    Path(str(base) + '_vocals.wav').write_bytes(b'separated audio')
                def transcribe_selected(source, _model, output, *args):
                    seen.append(Path(source))
                    self.assertEqual(Path(source).read_bytes(), b'separated audio')
                    midi = mido.MidiFile()
                    midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80),
                                                       mido.Message('note_off', note=60, time=480)]))
                    midi.save(output)
                arguments = [str(audio), '--model-path', 'Piano', '--output', str(root),
                             '--separate-stem', 'vocals', '--piano-separator', 'pymss', '--pymss-model', 'model']
                if keep:
                    arguments.append('--save-separated-stem')
                with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate), \
                     patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_selected):
                    result = CliRunner().invoke(transcribe, arguments)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue((root / 'song.mid').is_file())
                self.assertFalse(seen[0].exists())
                self.assertEqual((root / 'song_vocals.wav').exists(), keep)
                if keep:
                    self.assertEqual((root / 'song_vocals.wav').read_bytes(), b'separated audio')

    def test_saved_stem_is_never_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio, existing = root / 'song.wav', root / 'song_piano.wav'
            audio.touch()
            existing.write_bytes(b'keep')
            with patch('omnizart.desktop.six_stems.separate') as separator:
                result = CliRunner().invoke(transcribe, [str(audio), '--output', str(root),
                    '--piano-only', '--save-separated-stem'])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('Refusing to overwrite', result.output)
            self.assertEqual(existing.read_bytes(), b'keep')
            separator.assert_not_called()

    def test_piano_stem_can_be_kept_next_to_midi(self):
        import mido
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            def isolate(_source, base, _model, selected_stems, source_name):
                self.assertEqual(selected_stems, ['piano'])
                Path(str(base) + '_piano.wav').write_bytes(b'piano')
            def transcribe_piano(_source, _model, output, *args):
                midi = mido.MidiFile()
                midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80),
                                                   mido.Message('note_off', note=60, time=480)]))
                midi.save(output)
            with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate), \
                 patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_piano):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Piano', '--output', str(root),
                    '--piano-only', '--piano-separator', 'pymss', '--pymss-model', 'model', '--save-separated-stem'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual((root / 'song_piano.wav').read_bytes(), b'piano')
            self.assertTrue((root / 'song.mid').is_file())

    def test_pymss_isolation_multitrack_models_support_neural_score(self):
        import pretty_midi
        for model in ('YourMT3', 'MuScriptor'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                root = Path(directory)
                audio = root / 'song.wav'
                audio.touch()
                def isolate(_source, base, _model, selected_stems, source_name):
                    self.assertEqual(selected_stems, ['piano'])
                    Path(str(base) + '_piano.wav').touch()
                def transcribe_piano(_audio, _model, output, *args):
                    midi = pretty_midi.PrettyMIDI()
                    for program, pitch in ((0, 48), (40, 72)):
                        instrument = pretty_midi.Instrument(program=program)
                        instrument.notes.append(pretty_midi.Note(80, pitch, 0, 1))
                        midi.instruments.append(instrument)
                    midi.write(str(output))
                with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate), \
                     patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_piano), \
                     patch('omnizart.desktop.svsep.predict_staff', return_value=[2, 1]):
                    result = CliRunner().invoke(transcribe, [str(audio), '--model-path', model,
                        '--output', str(root), '--piano-only', '--piano-separator', 'pymss',
                        '--pymss-model', 'piano-model', '--score', '--neural-staff'])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue((root / 'song.musicxml').is_file())
                self.assertTrue((root / ('song_' + model.lower() + '_multitrack.mid')).is_file())

    def test_pymss_piano_isolation_feeds_selected_stem_to_transcription(self):
        import mido
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            observed = []
            def isolate(source, base, model, selected_stems, source_name):
                observed.append((source, model, selected_stems))
                Path(str(base) + '_piano.wav').touch()
            def transcribe_piano(source, model, output, *args):
                self.assertTrue(Path(source).name.endswith('_piano.wav'))
                midi = mido.MidiFile()
                midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80),
                                                   mido.Message('note_off', note=60, time=480)]))
                midi.save(output)
            with patch('omnizart.desktop.pymss_backend.separate', side_effect=isolate), \
                 patch('omnizart.desktop.six_stems.separate') as six, \
                 patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_piano):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Piano', '--output', str(root),
                    '--piano-only', '--piano-separator', 'pymss', '--pymss-model', 'my-piano', '--score'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(observed, [(str(audio), 'my-piano', ['piano'])])
            six.assert_not_called()
            self.assertTrue((root / 'song.mid').is_file())
            self.assertTrue((root / 'song.musicxml').is_file())

    def test_already_isolated_piano_multitrack_models_need_no_separator(self):
        import pretty_midi
        from xml.etree import ElementTree as ET
        for model in ('YourMT3', 'MuScriptor'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                root = Path(directory)
                audio = root / 'piano.wav'
                audio.touch()
                def make_output(_audio, _model, output, *args):
                    midi = pretty_midi.PrettyMIDI()
                    for program, pitch in ((0, 48), (40, 72)):
                        instrument = pretty_midi.Instrument(program=program)
                        instrument.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=0, end=1))
                        midi.instruments.append(instrument)
                    midi.write(str(output))
                with patch('omnizart.desktop.six_stems.separate') as separator, \
                     patch('omnizart.cli.music.transcribe._transcribe', side_effect=make_output), \
                     patch('omnizart.desktop.svsep.predict_staff', return_value=[2, 1]):
                    result = CliRunner().invoke(transcribe, [str(audio), '--model-path', model,
                        '--input-is-piano', '--output', str(root), '--score', '--neural-staff'])
                self.assertEqual(result.exit_code, 0, result.output)
                separator.assert_not_called()
                original = pretty_midi.PrettyMIDI(str(root / ('piano_' + model.lower() + '_multitrack.mid')))
                piano = pretty_midi.PrettyMIDI(str(root / 'piano.mid'))
                self.assertEqual(len(original.instruments), 2)
                self.assertEqual(len(piano.instruments), 1)
                self.assertEqual(sorted(n.pitch for n in piano.instruments[0].notes), [48, 72])
                score = ET.parse(root / 'piano.musicxml')
                self.assertEqual(score.findtext('.//staves'), '2')
                self.assertEqual(len(score.findall('.//note/pitch')), 2)

    def test_already_piano_rejects_separation(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / 'song.wav'
            audio.touch()
            result = CliRunner().invoke(transcribe, [str(audio), '--input-is-piano', '--piano-only', '--output', directory])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('cannot be combined', result.output)

    def test_already_piano_preserves_midi_extension_for_multi_instrument_model(self):
        import mido
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            def make_midi(_audio, _model, output, *args):
                midi = mido.MidiFile()
                midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80),
                                                   mido.Message('note_off', note=60, time=480)]))
                midi.save(output)
            with patch('omnizart.cli.music.transcribe._transcribe', side_effect=make_midi):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Pop',
                    '--input-is-piano', '--output', str(root / 'chosen.midi'), '--score'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((root / 'chosen.midi').is_file())
            self.assertTrue((root / 'chosen.musicxml').is_file())

    def test_piano_only_yourmt3_preserves_multitrack_and_writes_one_piano(self):
        import pretty_midi
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'song.wav'
            audio.touch()
            def make_output(_audio, _model, output, *args):
                midi = pretty_midi.PrettyMIDI()
                for program, pitch in ((0, 60), (40, 67)):
                    instrument = pretty_midi.Instrument(program=program)
                    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=0, end=1))
                    midi.instruments.append(instrument)
                midi.write(str(output))
            with patch('omnizart.desktop.six_stems.separate'), \
                 patch('omnizart.cli.music.transcribe._transcribe', side_effect=make_output):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'YourMT3',
                    '--piano-only', '--output', str(root), '--score'])
            self.assertEqual(result.exit_code, 0, result.output)
            piano = pretty_midi.PrettyMIDI(str(root / 'song.mid'))
            original = pretty_midi.PrettyMIDI(str(root / 'song_yourmt3_multitrack.mid'))
            self.assertEqual(len(piano.instruments), 1)
            self.assertEqual(piano.instruments[0].program, 0)
            self.assertEqual(sorted(note.pitch for note in piano.instruments[0].notes), [60, 67])
            self.assertEqual(len(original.instruments), 2)
            self.assertTrue((root / 'song.musicxml').is_file())

    def test_piano_simplification_writes_full_and_separate_score(self):
        import mido
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / 'song.wav'
            audio.touch()
            def make_midi(*args):
                midi = mido.MidiFile()
                midi.tracks.append(mido.MidiTrack([
                    mido.Message('note_on', note=60, velocity=80),
                    mido.Message('note_off', note=60, time=480)]))
                midi.save(str(folder / 'song.mid'))
            with patch('omnizart.cli.music.transcribe._transcribe', side_effect=make_midi):
                result = CliRunner().invoke(transcribe, [str(audio), '--model-path', 'Transkun (piano)',
                    '--output', str(folder), '--score', '--simplify-score'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((folder / 'song.musicxml').is_file())
            self.assertTrue((folder / 'song_simplified.musicxml').is_file())
            self.assertTrue((folder / 'song.mid').is_file())

    def test_optional_uvr_model_is_routed(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.wav"
            audio.touch()
            with patch("omnizart.desktop.sherpa_uvr.separate") as separator, \
                    patch("omnizart.cli.music.transcribe._transcribe"):
                result = CliRunner().invoke(transcribe, [str(audio), "--remove-vocals", "--separator", "inst-hq4"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(separator.call_args.args[0], "inst-hq4")

    def test_accompaniment_uses_the_explicit_run_name(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "song.mp3"
            audio.touch()
            (folder / "song_accompaniment.wav").write_bytes(b"keep")
            def create_stem(source, destination):
                destination.write_bytes(b"second")
            with patch("omnizart.desktop.separation.remove_vocals", side_effect=create_stem), \
                    patch("omnizart.cli.music.transcribe._transcribe"):
                result = CliRunner().invoke(transcribe, [str(audio), "--remove-vocals", "--save-accompaniment",
                                                       "--output", str(folder / "song_2")])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual((folder / "song_accompaniment.wav").read_bytes(), b"keep")
            self.assertEqual((folder / "song_2_accompaniment.wav").read_bytes(), b"second")

    def test_smoothing_saves_a_second_midi(self):
        import mido
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "song.wav"
            audio.touch()
            def make_midi(*args):
                midi = mido.MidiFile()
                midi.tracks.append(mido.MidiTrack([
                    mido.Message("note_on", note=60, velocity=80),
                    mido.Message("note_off", note=60, time=20),
                    mido.Message("note_on", note=60, velocity=80, time=1),
                    mido.Message("note_off", note=60, time=20)]))
                midi.save(str(folder / "song.mid"))
            with patch("omnizart.cli.music.transcribe._transcribe", side_effect=make_midi):
                result = CliRunner().invoke(transcribe, [str(audio), "--smooth-notes", "--output", str(folder)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((folder / "song.mid").is_file())
            cleaned = mido.MidiFile(str(folder / "song_smoothed.mid"))
            self.assertEqual(sum(m.type == "note_on" for t in cleaned.tracks for m in t), 1)

    def test_uvr_gpu_and_index_are_forwarded(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.wav"
            audio.touch()
            with patch("omnizart.desktop.mdx.separate") as separation, \
                    patch("omnizart.cli.music.transcribe._transcribe"):
                result = CliRunner().invoke(transcribe, [str(audio), "--remove-vocals", "--separator", "uvr",
                                                       "--uvr-device", "directml", "--gpu-index", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(separation.call_args.kwargs, {"device": "directml", "gpu_index": 1})

    def test_gpu_is_rejected_for_spleeter(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.wav"
            audio.touch()
            result = CliRunner().invoke(transcribe, [str(audio), "--remove-vocals", "--uvr-device", "directml"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("DirectML requires piano-only or UVR vocal separation", result.output)

    def test_uvr_is_selected_and_passes_both_stems(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.wav"
            audio.touch()
            with patch("omnizart.desktop.mdx.separate") as separation, \
                    patch("omnizart.cli.music.transcribe._transcribe") as run, \
                    patch("omnizart.desktop.combined.add_vocals") as combine:
                result = CliRunner().invoke(transcribe, [str(audio), "--remove-vocals", "--include-vocals",
                                                       "--separator", "uvr", "--output", directory])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(separation.call_args.args[0], str(audio))
            self.assertEqual(separation.call_args.args[2].name, "vocals.wav")
            self.assertEqual(run.call_args.args[0], str(separation.call_args.args[1]))
            combine.assert_called_once()

    def test_separated_audio_does_not_overwrite_existing_file(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "song.mp3"
            audio.touch()
            existing = folder / "song_accompaniment.wav"
            existing.write_bytes(b"keep")
            def create_stem(source, destination):
                destination.write_bytes(b"new")
            with patch("omnizart.desktop.separation.remove_vocals", side_effect=create_stem), \
                    patch("omnizart.cli.music.transcribe._transcribe") as run:
                result = CliRunner().invoke(transcribe, [str(audio), "--remove-vocals", "--save-accompaniment",
                                                       "--output", str(folder)])
            self.assertNotEqual(result.exit_code, 0)
            self.assertEqual(existing.read_bytes(), b"keep")
            run.assert_not_called()

    def test_separation_uses_a_temporary_file_and_preserves_name(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.touch()
            with patch("omnizart.desktop.separation.remove_vocals") as separation, \
                    patch("omnizart.cli.music.transcribe._transcribe") as run:
                result = CliRunner().invoke(transcribe, [str(audio), "--remove-vocals", "--model-path", "Piano"])
            self.assertEqual(result.exit_code, 0, result.output)
            temporary_audio = separation.call_args.args[1]
            self.assertEqual(temporary_audio.name, "song.wav")
            self.assertEqual(run.call_args.args[0], str(temporary_audio))
            self.assertFalse(temporary_audio.exists())


if __name__ == "__main__":
    unittest.main()

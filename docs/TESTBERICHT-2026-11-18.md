# Test report: Windows portable build, 2026-11-18

## Scope and build

The application source was commit `a3ac4b9`, which combines YourMT3
predictions into a single piano track when six-stem piano isolation is
selected. The original YourMT3 multitrack MIDI is retained separately. The
portable build was assembled with `scripts/windows/build_portable.py` and the
existing `.runtime311` environment. Its dependency substitutions and provenance
are recorded in `LOCAL-BUILD-NOTES.md` within the local build. No installer was
built because Inno Setup was unavailable.

## Results

- `python -m pip check`: passed in both the source and portable runtimes.
- `python -m unittest discover -s tests -p 'test_*.py' -v`: 113 test entries,
  including 103 successful unittest cases and 10 module import errors because
  `pytest` is absent. This is not a complete pytest run; pytest-style functions
  and parametrized tests were not executed. The previous stale DirectML error
  assertion was corrected before this run.
- Targeted notation, desktop command, beat grid, comparison and separation CLI
  suite: 44 tests passed before the final extension to the YourMT3 regression.
  The extended regression subsequently passed separately, including creation of
  a single piano MIDI, preservation of the original multitrack MIDI and MusicXML
  export. Six-stem separation and YourMT3 inference were mocked in this test.
- Portable GUI transcription smoke test: passed using the Piano model and the
  bundled test audio; the generated MIDI contained notes. This does not test
  YourMT3 or six-stem inference.
- `pytest` and `pytest-mock` were subsequently installed in an isolated test
  directory. Collection found 182 tests. The full pytest run reached at least
  74% but was interrupted when the execution environment restarted; it did not
  produce a final result. A separate check confirmed that
  `tests/feature/test_wrapper_func.py::test_extract_patch_cqt` fails because
  `tests/resource/sample_feature.hdf` is missing.
- A separate selection of 72 pytest tests reported 71 passed and one failure:
  `test_stem_mode_uses_optional_gpu_and_reports_wav_results` could not initialize
  Tcl in that combined run. The same test passed when run on its own. This
  order/environment-dependent behavior remains unresolved.

## Follow-up: complete pytest run

After the initial interruption, a diagnostic run completed all 182 tests, with
180 passing and two failures: the missing CQT reference fixture and a Tk/Tcl
initialization error under pytest's default file-descriptor capture. The latter
was reproducible with the desktop test module under default capture, while the
same tests passed with `--capture=sys` and with capture disabled. The project
now defaults to `--capture=sys` for pytest; this addresses the observed Tcl
test failure, not a proven cause of the earlier execution-host interruption.

The official `resource.zip` from the upstream `checkpoints-20211001` release
supplied `resource/sample_feature.hdf` to `tests/resource/sample_feature.hdf`.
The fixture is ignored by Git. When absent, the affected regression test now
reports an explicit skip; it still runs when the reference file is available.

With those test changes (commit `8a1ab3f`) and the reference fixture present,
`python -m pytest tests -q --disable-warnings -rs --tb=short` completed with
**182 passed, 11 subtests passed, 13 warnings** in 144.96 seconds. No tests
were skipped. `python -m pip check` also passed. The original host restart did
not recur, so its underlying cause remains unconfirmed.

## Limitations

A first request to install `pytest` and `pytest-mock` was not executed because
the sandbox's automatic approval review reported a usage limit; installation
into an isolated test directory succeeded later. Six-stem model
weights and the optional YourMT3 runtime were not installed, so their real
combined inference was not tested. GPU inference, fresh installation, upgrades
and uninstallation were not tested. The single-piano output merges predicted
notes and does not establish that every note represents a piano sound.

Adaptive rhythm and tuplets, a score editor, a curated reference set and a real
DuET/PM2S comparison remain unimplemented. These tests therefore do not establish
that all concepts discussed for score quality are complete or that a generated
score is correct or playable.

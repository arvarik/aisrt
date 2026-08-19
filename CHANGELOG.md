# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Standards-compliant subtitle formatting. Cues now follow the Netflix Timed
  Text Style Guide: 42 characters per line, 2 lines, 20 characters per second,
  a minimum duration of 5/6 second, a maximum of 7 seconds, and a minimum gap of
  two frames between cues.
- Balanced line breaking that never splits an article, a preposition, or an
  auxiliary verb from the word it belongs to.
- Graceful shutdown. `SIGINT` and `SIGTERM` stop the producer, drain the queues,
  flush the database, and exit with a clear status.
- `--version`, `--db-path`, `--force-compute-type`, `--language`, `--ext`,
  `--exclude`, and `--dry-run` command line options.
- A single `ffprobe` call per file now returns the duration, the audio tracks,
  and the embedded subtitle languages. Discovery and extraction share the result.
- Bounded concurrent probing during discovery.
- A memory budget for in-flight audio. The pipeline caps resident audio in bytes
  instead of in file count.
- `py.typed` marker, so type checkers use the annotations of the installed package.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, issue and pull request
  templates, Dependabot configuration, and a pre-commit configuration.
- Integration tests that run the real `ffmpeg` and `ffprobe` binaries.

### Changed

- Audio extraction writes into a preallocated NumPy buffer sized from the probed
  duration. Peak memory per file drops by about one third.
- The state database keeps history across upgrades. Schema changes now use
  `PRAGMA user_version` and `ALTER TABLE` instead of dropping the table.
- Directory scanning is iterative and streams results, so the first file starts
  processing before the crawl finishes.
- Exclude patterns match every path component and ignore case.
- The hardware router validates the compute type against CTranslate2 and reports
  honestly that Apple Silicon runs on the CPU.
- Environment variables such as `AISRT_TRANSLATE` now take effect. Command line
  options override them only when the user passes them.
- The Docker image builds in two stages, runs as an unprivileged user, and uses a
  CUDA 12.9 base with cuDNN 9, which CTranslate2 4.5 and later require.

### Fixed

- Subtitle lines overflowed the 42-character limit by up to 24 percent because
  the character counter ignored the spaces between words.
- Abbreviations such as `Mr.` and decimals such as `3.5` split a sentence and
  produced orphan cues.
- `_format_timestamp` produced a malformed timestamp for a negative time.
- The atomic writer did not flush the file before the rename, so a power loss
  could leave an empty subtitle file.
- A non-permission `OSError` during ownership inheritance discarded a finished
  transcription.
- The temporary subtitle file was world readable before the permission bits were
  applied.
- Every documented `AISRT_*` environment variable was ignored.
- The command exited with status 0 even when every file failed.
- `aisrt.__version__` reported a stale hardcoded value.
- A cue could appear up to a second before the words were spoken, because
  stretching it to a readable length moved the in-time backwards without bound.
- A single token longer than one cue collapsed the whole sentence into one cue.
- Closing a gap between cues could push one past the seven second maximum.
- The retry budget never took effect. Recording progress reset the attempt
  count, so a file that always failed was retried on every run forever.
- Passing one hardware option silenced every `AISRT_HARDWARE__*` variable,
  because a constructed nested model replaces the whole section instead of
  merging with it.
- `aisrt --version` emitted terminal colour codes, so a script could not parse it.
- Symlinked media files were measured by the link rather than the target, and
  named pipes, sockets, and broken symlinks entered the pipeline.
- Scanning `/media/movies` also loaded database rows for `/media/movies2`.
- Two tasks sharing the connection could interleave inside one transaction.
- A short `os.write` committed a truncated subtitle.
- A long release name made the temporary file exceed the filename limit.
- A failure after the rename was reported as a write failure for a file that had
  in fact been written correctly.
- The memory budget counted the visible slice rather than the buffer that stays
  resident behind it, so it under-counted by up to half.
- Shutting down during a transcription raised a confusing error. The worker now
  stops at the next segment boundary instead.


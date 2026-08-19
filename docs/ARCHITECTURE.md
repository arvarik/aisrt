# aisrt Architecture

**Status:** Describes the shipped implementation.

## 1. What the program does

`aisrt` crawls a media directory, decides which videos need a subtitle, decodes their
audio into memory, transcribes it, and writes a standards-compliant `.srt` file next to
each video.

The design goals, in priority order:

1. **Do not lose work.** A crash, a signal, or a corrupt file must never cost more than
   the file being processed at that moment.
2. **Do not corrupt anything.** The source video is never modified. A subtitle appears
   whole or not at all.
3. **Stay inside a memory budget.** A library of three-hour films must not push the
   process into the out-of-memory killer.
4. **Be accurate.** Cues follow the Netflix Timed Text Style Guide, and the model is
   configured to avoid the failure modes that long-form audio provokes.

## 2. Tech stack

| Concern | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | `asyncio.TaskGroup` and `asyncio.timeout` |
| CLI | Typer, Rich | Typed options, a readable terminal report |
| Logging | Loguru, to stderr | Keeps the progress bar on stdout uncorrupted |
| Configuration | Pydantic v2, pydantic-settings | Validation at startup, `AISRT_` variables |
| Speech to text | faster-whisper 1.2 (CTranslate2) | Fast, no PyTorch dependency |
| Media | `ffmpeg` and `ffprobe` as child processes | No Python codec bindings to keep current |
| State | aiosqlite | One file, no server, asynchronous |
| Hardware | CTranslate2 device query, `nvidia-ml-py`, psutil | Authoritative on what will actually load |

## 3. Module map

| Module | Responsibility |
| --- | --- |
| `cli.py` | Parses options, wires everything together, owns the progress display and signals |
| `config.py` | Pydantic models, validation, `AISRT_` variables |
| `hardware.py` | Reads the machine, routes to a model, device, and precision |
| `discovery.py` | Crawls the tree and decides what each file needs |
| `probing.py` | One `ffprobe` call per file, language normalisation, audio track choice |
| `extractor.py` | Decodes audio into a preallocated NumPy buffer |
| `stt.py` | Owns the model and the inference thread, builds the transcribe options |
| `pipeline.py` | Bounded producer and consumer stages, the memory budget |
| `assembly.py` | Turns word timestamps into cues, writes them atomically |
| `state.py` | The SQLite store, its schema, and its migrations |

## 4. The pipeline

Three stages run concurrently inside one `asyncio.TaskGroup`.

```
                  extraction_queue          inference_queue
producer  ────────────────────────►  extractor  ────────────►  inference
(crawl +  maxsize = 12               x3 workers  maxsize = 2   1 worker
 probe)                              (I/O bound)               (model thread)
```

- **Producer.** Streams files out of the crawler as they are found, so extraction starts
  before the crawl finishes. It stops queueing when the shutdown event is set.
- **Extractors.** Three workers, because extraction is I/O bound and inference is the
  bottleneck. Each reserves memory from the budget before it decodes.
- **Inference.** One worker, feeding one thread that owns the model. Serializing here is
  what keeps VRAM usage predictable.

Shutdown is ordered: the producer finishes and sends one sentinel per extractor, every
extractor drains and exits, then the inference worker receives its own sentinel.

### 4.1 The memory budget

Bounding the queues by item count is not enough. Three minutes of audio and three hours of
audio occupy the same queue slot but differ sixtyfold in memory. `MemoryBudget` counts
bytes instead. An extractor reserves an estimate from the probed duration, corrects the
reservation once the real size is known, and the inference worker releases it after the
subtitle is written. A file larger than the whole budget is admitted alone rather than
deadlocking the pipeline.

Default budget: 2048 MB, adjustable with `--max-memory-mb`.

### 4.2 Failure isolation

A corrupt file marks itself `FAILED` and the pipeline continues. State writes are wrapped
so that a locked database cannot take the run down. Cancellation is re-raised before the
broad exception handler, so a shutdown is never mistaken for a file failure.

## 5. Discovery

The crawl is an explicit-stack iterator, so a deep tree cannot exhaust the recursion
limit, and it yields files as it finds them rather than building a full list first.

- Exclude patterns are matched with `fnmatch` against each lowercased path component. The
  earlier implementation used `Path.match`, which is case-sensitive and only inspects the
  final component, so `Sample` and `Extras` directories were never excluded.
- One `lstat` per media file supplies size, mtime, device, and inode. Nothing else is
  stat'd.
- A failure on one entry or one directory is logged and skipped. A stale NFS handle no
  longer aborts a subtree.
- Symlinked directories are skipped unless `follow_symlinks` is set, in which case loops
  are detected by device and inode.

Each candidate is then probed, with at most four probes running at once.

### 5.1 Skip reasons

In order: modified too recently, a sidecar subtitle exists, a database row says the work
is done, the file failed too many times, the content is a hardlink of something finished,
the probe failed, there is no audio track, or the container already carries a text
subtitle.

## 6. Probing

One `ffprobe` call per file returns the duration, every audio track, and every subtitle
language. The earlier implementation ran two calls, one during discovery and one during
extraction, each reading up to 5 MB of the file. Merging them halves the metadata reads
over the network and supplies the duration that sizes the audio buffer.

- Language tags are normalised: `eng`, `en`, `EN`, and `en-US` all become `en`.
- The audio track is chosen by preferred language, then the container's default
  disposition, then channel count. When translating, no language is preferred, because the
  original-language track is the useful one.
- Image subtitles such as PGS do not count as an existing subtitle, because they force a
  transcode on many players.
- A failed probe is reported as `probe_failed`, never as "this file has no subtitles".
- Every call has a deadline and runs in its own process group, so a stalled mount cannot
  hang the run.

## 7. Zero-disk extraction

```
ffmpeg -nostdin -hide_banner -loglevel error
       -fflags +discardcorrupt -err_detect crccheck
       -threads 1
       -i <video> -map 0:a:<n> -vn -sn -dn
       -ac 1 -ar 16000 -f s16le -max_error_rate 0.5 -
```

The raw PCM is scaled directly into a `float32` buffer sized from the probed duration. The
earlier implementation grew a `bytearray` and then called `.astype(np.float32)`, which
held both the byte buffer and the converted array at once: about 710 MB peak for a
two-hour film against 461 MB now, a 35 percent reduction.

Robustness details that matter in practice:

- `-nostdin` plus `stdin=DEVNULL`. Without them a background process group can stop
  FFmpeg with `SIGTTIN`, which looks exactly like a hang.
- `start_new_session=True` so the whole process group can be signalled.
- `asyncio.timeout`, then `SIGTERM`, then `SIGKILL`, each with its own deadline. The
  cleanup never uses `communicate()`, which waits for pipe EOF rather than process exit
  and can therefore block forever after a kill.
- A one-byte carry across chunk boundaries, because a 16-bit sample can straddle a pipe
  read.

## 8. Transcription

Two decoding paths exist, and they honour different arguments.

**Sequential** is the default. It keeps the temperature fallback ladder, the compression
ratio and log probability thresholds, and `hallucination_silence_threshold`. Those are
what break a repetition loop on a film with long silent stretches.

**Batched** is roughly three times faster and is enabled with `--batch-size`.
`BatchedInferencePipeline` accepts and then discards `temperature`,
`compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`,
`condition_on_previous_text`, and `hallucination_silence_threshold`. `build_transcribe_options`
does not pass them there, because doing so implies a guard that is not running.

Other decisions:

- **No `initial_prompt`.** A style prompt biases the model, and in batched mode it is
  re-injected into every chunk and can appear in the transcript.
- **`condition_on_previous_text=False`.** Carrying text across a long musical passage is
  the classic trigger for a repeated caption.
- **Language detection first.** `detect_language` runs the encoder only, with the voice
  activity filter on, so it reads dialogue rather than the studio logo. The detected code
  is then pinned for the whole file.
- **Turbo cannot translate.** The turbo checkpoint and every `*.en` checkpoint return the
  original language when asked to translate, so `--translate` routes to `large-v3` or
  `medium`.

## 9. Subtitle assembly

Word timestamps become cues in seven ordered passes, each owning one invariant.

1. **Flatten and sanitise.** Words from every segment become one stream; times are clamped
   and forced to move forward. Whisper's own segment boundaries come from decoder
   chunking, not prosody, so they are discarded.
2. **Segment.** Cut after a sentence, and at any silence of half a second or more.
   Abbreviations (`Mr.`), initials (`A.`), and decimals (`3.5`) do not end a sentence.
3. **Split.** Break an oversized sentence at the most natural point: a sentence end, then
   a clause end, then the largest silence, then before a conjunction.
4. **Merge.** Join neighbouring cues while the result stays inside every limit. This is
   what removes one-word cues below the minimum duration.
5. **Wrap.** Lay each cue over at most two balanced lines. A line never ends on an
   article, preposition, conjunction, or auxiliary verb.
6. **Retime.** Give each cue a readable duration inside the free time available, honouring
   the reading speed, the minimum duration, and the maximum.
7. **Normalise gaps.** Force every neighbouring pair apart by exactly the minimum gap. An
   overlap makes a player drop a cue.

| Limit | Value | Source |
| --- | --- | --- |
| Characters per line | 42 | Netflix English Timed Text Style Guide |
| Lines per cue | 2 | same |
| Reading speed | 20 characters per second | same |
| Minimum duration | 5/6 second | Netflix General Requirements |
| Maximum duration | 7 seconds | same |
| Gap between cues | 0.084 second, two frames | Netflix Subtitle Timing Guidelines |

Output is UTF-8 without a byte order mark, LF line endings, one blank line between cues,
and a single trailing newline.

### 9.1 Atomic writing

1. Write to a hidden temporary file in the same directory, created with mode `0600`.
2. `fsync` the file so a power loss cannot leave it empty.
3. Copy the owner and permission bits from the source video, minus the execute bits. A
   failure here is logged, never fatal: losing a finished transcription because a network
   share refuses `chown` would be a poor trade.
4. `os.replace` onto the final name. Same directory, so the rename is atomic.
5. `fsync` the directory so the rename itself survives a power loss.

The sidecar name is built by joining components, never by `Path.with_suffix`, which would
eat the last dotted component of an extension-less filename.

## 10. State

One SQLite file, opened with write-ahead logging, `synchronous=NORMAL`, a busy timeout,
and a 64 MB page cache. The result of the journal mode pragma is checked, because SQLite
silently falls back to rollback journalling on a filesystem without shared memory, which
is every network mount.

Migrations use `PRAGMA user_version` and `ALTER TABLE`. They never drop a table. An
earlier build dropped `file_state` on upgrade, which destroyed a user's entire history.

The scan loads only the columns it reads, filtered to the media directory, so a shared
database does not pull in rows for another library. Discovery batches its writes; the
pipeline commits each transition, because a durable transition is what makes a crash
recoverable.

## 11. Shutdown

`SIGINT` and `SIGTERM` are handled on the event loop, not by an interrupt at an arbitrary
bytecode boundary.

- **First signal:** set the stop event. The producer stops queueing, in-flight files
  finish, their state commits, and the database closes. Watch mode wakes from its sleep
  immediately rather than waiting out the interval.
- **Second signal:** cancel every task and exit.

The process exits `130` after a signal, `1` if any file failed, `2` on a configuration
error, and `0` otherwise.

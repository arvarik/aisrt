# Security Policy

## Supported Versions

Security fixes land on the latest released version. Please upgrade before you
report a problem.

| Version | Supported |
| ------- | --------- |
| Latest release | Yes |
| Older releases | No |

## Reporting a Vulnerability

Do not open a public issue for a security problem.

Use GitHub private vulnerability reporting instead:
[Report a vulnerability](https://github.com/arvarik/aisrt/security/advisories/new).

Include the following information:

1. The version of `aisrt` and the operating system.
2. The exact steps that reproduce the problem.
3. What an attacker gains if the problem is not fixed.

We acknowledge every report within 5 working days. We aim to publish a fix
within 30 days for a confirmed high-severity problem.

## Threat Model

`aisrt` runs as a local daemon. It reads media files, runs `ffmpeg` and
`ffprobe` as child processes, downloads AI models from Hugging Face, and writes
subtitle files next to the source media.

These areas carry the most risk:

- **Subprocess arguments.** File paths reach `ffmpeg` and `ffprobe`. The code
  always uses `asyncio.create_subprocess_exec` with an argument list, never a
  shell string, so a path cannot inject a command.
- **Model downloads.** The first run downloads model weights from Hugging Face.
  Set `HF_TOKEN` to use an authenticated download. Pin a local model directory
  with `--force-model /path/to/model` if you need an offline install.
- **File writes.** The tool writes only `.srt` files and a temporary file in the
  same directory. It never modifies the source video.
- **Ownership inheritance.** The tool copies the owner and the permission bits
  of the source video onto the subtitle file. Run the daemon as the user that
  owns the media library, not as root.

## Out of Scope

- Problems in `ffmpeg`, `faster-whisper`, or `CTranslate2`. Report those
  upstream.
- The accuracy of a generated transcript.

"""Tests for media probing."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aisrt.probing import (
    AudioTrack,
    FFmpegNotFoundError,
    MediaInfo,
    external_subtitle_path,
    has_external_subtitle,
    normalize_language,
    normalize_languages,
    probe_media,
    require_ffmpeg,
)
from tests.conftest import ffmpeg_installed

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_installed(), reason="FFmpeg is not installed")


class TestLanguageNormalisation:
    """Container language tags arrive in many shapes."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("eng", "en"),
            ("en", "en"),
            ("EN", "en"),
            (" en-US ", "en"),
            ("en_GB", "en"),
            ("fra", "fr"),
            ("fre", "fr"),
            ("jpn", "ja"),
            ("qqq", "qqq"),
            ("und", None),
            ("", None),
            (None, None),
        ],
    )
    def test_normalize(self, raw: str | None, expected: str | None) -> None:
        """Each tag reduces to a lowercase two-letter code where one exists."""
        assert normalize_language(raw) == expected

    def test_normalize_list(self) -> None:
        """A list of aliases collapses to one code."""
        assert normalize_languages(["eng", "en", "und", "fra"]) == frozenset({"en", "fr"})


class TestAudioTrackSelection:
    """The chosen audio track drives the whole transcript."""

    @staticmethod
    def _info(*tracks: AudioTrack) -> MediaInfo:
        return MediaInfo(duration=100.0, audio_tracks=tracks)

    def test_prefers_the_target_language(self) -> None:
        """A track in a wanted language wins over the default track."""
        info = self._info(
            AudioTrack(0, "fr", "ac3", 6, True, None),
            AudioTrack(1, "en", "aac", 2, False, None),
        )
        assert info.select_audio_track(frozenset({"en"})) == 1

    def test_falls_back_to_the_default_track(self) -> None:
        """With no language match, the container's default track is used."""
        info = self._info(
            AudioTrack(0, "ja", "aac", 2, False, "Commentary"),
            AudioTrack(1, "ja", "ac3", 6, True, "Main"),
        )
        assert info.select_audio_track(frozenset({"en"})) == 1

    def test_falls_back_to_the_widest_track(self) -> None:
        """With no default either, the track with the most channels wins."""
        info = self._info(
            AudioTrack(0, None, "aac", 2, False, None),
            AudioTrack(1, None, "dts", 8, False, None),
        )
        assert info.select_audio_track(frozenset({"en"})) == 1

    def test_translation_ignores_the_language_preference(self) -> None:
        """Translating needs the original audio, so no language is preferred."""
        info = self._info(
            AudioTrack(0, "ja", "ac3", 6, True, None),
            AudioTrack(1, "en", "aac", 2, False, None),
        )
        assert info.select_audio_track(frozenset()) == 0

    def test_no_audio_returns_zero(self) -> None:
        """A file with no audio still returns a usable index."""
        assert MediaInfo(duration=None).select_audio_track(frozenset({"en"})) == 0


class TestProbeMedia:
    """probe_media turns one ffprobe document into a MediaInfo."""

    @staticmethod
    def _document() -> dict[str, object]:
        return {
            "format": {"duration": "5400.5"},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264"},
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "ac3",
                    "channels": 6,
                    "disposition": {"default": 1},
                    "tags": {"language": "eng"},
                },
                {
                    "index": 2,
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "tags": {"language": "fre"},
                },
                {
                    "index": 3,
                    "codec_type": "subtitle",
                    "codec_name": "hdmv_pgs_subtitle",
                    "tags": {"language": "eng"},
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_parses_every_field(self) -> None:
        """Duration, audio tracks, and text subtitles all come from one call."""
        with patch("aisrt.probing._run_ffprobe", new=AsyncMock(return_value=self._document())):
            info = await probe_media(Path("movie.mkv"))

        assert info.duration == pytest.approx(5400.5)
        assert len(info.audio_tracks) == 1
        assert info.audio_tracks[0].language == "en"
        assert info.audio_tracks[0].channels == 6
        assert info.audio_tracks[0].is_default is True
        assert info.probe_failed is False

    @pytest.mark.asyncio
    async def test_image_subtitles_do_not_count(self) -> None:
        """A PGS track forces a transcode, so it is not a usable subtitle."""
        with patch("aisrt.probing._run_ffprobe", new=AsyncMock(return_value=self._document())):
            info = await probe_media(Path("movie.mkv"))

        assert info.has_text_subtitle(frozenset({"en"})) is False
        assert info.has_text_subtitle(frozenset({"fr"})) is True

    @pytest.mark.asyncio
    async def test_a_failed_probe_is_reported_not_hidden(self) -> None:
        """An empty document must not read as "this file has no subtitles"."""
        with patch("aisrt.probing._run_ffprobe", new=AsyncMock(return_value={})):
            info = await probe_media(Path("broken.mkv"))

        assert info.probe_failed is True
        assert info.duration is None
        assert info.audio_tracks == ()

    @pytest.mark.asyncio
    async def test_a_missing_duration_is_none(self) -> None:
        """A container without a duration header does not raise."""
        document = {"format": {}, "streams": [{"codec_type": "audio", "codec_name": "aac"}]}
        with patch("aisrt.probing._run_ffprobe", new=AsyncMock(return_value=document)):
            info = await probe_media(Path("stream.ts"))

        assert info.duration is None
        assert len(info.audio_tracks) == 1


class TestExternalSubtitle:
    """A sidecar subtitle keeps a file out of the pipeline."""

    def test_finds_the_plain_name(self, tmp_path: Path) -> None:
        """``movie.srt`` counts."""
        (tmp_path / "movie.mkv").write_text("v")
        (tmp_path / "movie.srt").write_text("s")
        assert has_external_subtitle(tmp_path / "movie.mkv", ["en"]) is True

    @pytest.mark.parametrize("suffix", ["en", "eng"])
    def test_finds_language_variants(self, tmp_path: Path, suffix: str) -> None:
        """Both the two-letter and the three-letter forms count."""
        (tmp_path / "movie.mkv").write_text("v")
        (tmp_path / f"movie.{suffix}.srt").write_text("s")
        assert has_external_subtitle(tmp_path / "movie.mkv", ["en"]) is True

    def test_ignores_another_language(self, tmp_path: Path) -> None:
        """A French sidecar does not satisfy an English target."""
        (tmp_path / "movie.mkv").write_text("v")
        (tmp_path / "movie.fr.srt").write_text("s")
        assert has_external_subtitle(tmp_path / "movie.mkv", ["en"]) is False

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        """No sidecar means no path."""
        (tmp_path / "movie.mkv").write_text("v")
        assert external_subtitle_path(tmp_path / "movie.mkv", frozenset({"en"})) is None

    def test_matches_a_dotted_release_name(self, tmp_path: Path) -> None:
        """The writer and the checker agree on a dotted name."""
        video = tmp_path / "Movie.2019.1080p.mkv"
        video.write_text("v")
        (tmp_path / "Movie.2019.1080p.en.srt").write_text("s")
        assert has_external_subtitle(video, ["en"]) is True


class TestRequireFFmpeg:
    """A missing toolchain must fail once, at startup."""

    def test_passes_when_present(self) -> None:
        """Both binaries on PATH means no error."""
        with patch("aisrt.probing.shutil.which", return_value="/usr/bin/ffmpeg"):
            require_ffmpeg()

    def test_raises_when_missing(self) -> None:
        """A missing binary names itself in the message."""
        with (
            patch("aisrt.probing.shutil.which", return_value=None),
            pytest.raises(FFmpegNotFoundError, match="ffmpeg and ffprobe not found"),
        ):
            require_ffmpeg()


@needs_ffmpeg
@pytest.mark.integration
class TestProbeIntegration:
    """These tests run the real ffprobe binary."""

    @staticmethod
    def _make_media(path: Path, duration: float = 2.0) -> None:
        """Build a small file with one audio track and one subtitle track."""
        subtitle = path.with_suffix(".srt")
        # The subtitle must span the whole clip, because -shortest truncates the
        # output to the shortest mapped stream.
        subtitle.write_text(
            f"1\n00:00:00,000 --> 00:00:{duration:06.3f}\nHello.\n".replace(".", ",", 1),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={duration}",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s=64x64:d={duration}",
                "-i",
                str(subtitle),
                "-map",
                "1:v",
                "-map",
                "0:a",
                "-map",
                "2:s",
                "-metadata:s:a:0",
                "language=eng",
                "-metadata:s:s:0",
                "language=eng",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                "-c:s",
                "srt",
                "-shortest",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        subtitle.unlink()

    @pytest.mark.asyncio
    async def test_probes_a_real_file(self, tmp_path: Path) -> None:
        """A real container yields its duration, audio track, and subtitle."""
        media = tmp_path / "sample.mkv"
        self._make_media(media)

        info = await probe_media(media)

        assert info.probe_failed is False
        assert info.duration == pytest.approx(2.0, abs=0.3)
        assert len(info.audio_tracks) == 1
        assert info.audio_tracks[0].language == "en"
        assert info.has_text_subtitle(frozenset({"en"})) is True

    @pytest.mark.asyncio
    async def test_a_corrupt_file_reports_failure(self, tmp_path: Path) -> None:
        """Garbage on disk fails the probe instead of raising."""
        broken = tmp_path / "broken.mkv"
        broken.write_bytes(b"this is not a media file" * 100)

        info = await probe_media(broken)

        assert info.probe_failed is True

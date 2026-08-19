# Homebrew formula for aisrt.
#
#   brew tap arvarik/aisrt https://github.com/arvarik/aisrt
#   brew install aisrt
#
# KNOWN LIMITATION: Homebrew builds run inside a network sandbox, so pip cannot
# fetch dependencies at build time. This formula therefore needs a `resource`
# block for every dependency, generated with:
#
#   brew update-python-resources Formula/aisrt.rb
#
# Until those blocks exist, install with `uv tool install aisrt` or
# `pipx install aisrt` instead.

class Aisrt < Formula
  include Language::Python::Virtualenv

  desc "Hardware-aware pipeline for broadcast-quality subtitle generation"
  homepage "https://github.com/arvarik/aisrt"
  url "https://files.pythonhosted.org/packages/source/a/aisrt/aisrt-1.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # updated by CI
  license "Apache-2.0"

  depends_on "python@3.13"
  # aisrt shells out to both binaries and refuses to start without them.
  depends_on "ffmpeg"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/aisrt --version")
    assert_match "scan", shell_output("#{bin}/aisrt --help")
  end
end

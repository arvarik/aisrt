# Homebrew formula for aisrt
# Version is dynamically determined from PyPI — no hardcoded versions.
#
# Usage:
#   brew tap arvarik/aisrt https://github.com/arvarik/aisrt
#   brew install aisrt
#
# This formula creates a Python virtualenv and installs aisrt via pip.

class Aisrt < Formula
  include Language::Python::Virtualenv

  desc "Hardware-aware, concurrent pipeline for subtitle generation"
  homepage "https://github.com/arvarik/aisrt"
  url "https://files.pythonhosted.org/packages/source/a/aisrt/aisrt-1.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # Placeholder, will be updated by GH action
  license "Apache-2.0"

  depends_on "python@3.11"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "aisrt", shell_output("#{bin}/aisrt --help")
  end
end

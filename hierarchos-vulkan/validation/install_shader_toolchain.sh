#!/usr/bin/env bash
# Install the pinned shader toolchain into .tools/ without administrator rights.
#
# The tree's provenance rule is that the pinned releases reproduce every committed
# artifact byte-for-byte, and these are the releases that do (467 pinned artifacts
# with `validation/check_shader_provenance.py`):
#
#   glslangValidator  16.5.0            official Khronos glslang release, Windows x86-64
#   glslangValidator  16.6.0            the same project's next release; five artifacts
#                                       (silu, both GELU-tanh modules, Kimi softmax and
#                                       the routed-MoE router) only reproduce here
#   glslc             shaderc 2025.3    conda-forge package; the release it reports
#                                       about itself is 2023.8, which is what the
#                                       manifest records per artifact
#   spirv-val         SPIRV-Tools       conda-forge package, validation only
#
# Each manifest entry records the release that reproduces it and the checker
# selects that release per artifact, so the pins below are what a rebuild verifies
# rather than what a flag has to name. The checker discovers both glslang releases
# under .tools/, so naming them on the command line is unnecessary.

# Neither SPIRV-Tools nor shaderc publishes Windows binaries of its own, so those
# two come from conda-forge packages unpacked by micromamba. Nothing is installed
# system-wide: the tools live in .tools/ (gitignored) and no path outside this
# repository is modified.
#
# Sourcing .tools/env.sh puts the pinned glslang directory first on PATH, so the
# name `glslangValidator` resolves to the pinned release rather than to the
# glslang that the shaderc package depends on (15.4.0 for shaderc 2025.3; the
# 2026.4 package pulls 16.6.0, which does not reproduce the attention kernels).
#
# Usage, from hierarchos-vulkan/:
#
#   bash validation/install_shader_toolchain.sh
#   source ../.tools/env.sh
#   python validation/check_shader_provenance.py
#   python validation/verify_training_shaders.py
#
# Add `--record-versions` to the first command after rebuilding an artifact under a
# different compiler release: it writes the release that reproduced the artifact
# back into the manifest.
#
# Re-running is a no-op for anything already installed; delete .tools/ to start over.
set -euo pipefail

GLSLANG_VERSION="16.5.0"
GLSLANG_URL="https://github.com/KhronosGroup/glslang/releases/download/${GLSLANG_VERSION}/glslang-${GLSLANG_VERSION}-windows-x86_64-release.zip"
# Published SHA-256 of that release asset; the download is rejected if it differs.
GLSLANG_SHA256="06b71298b750268c127f2ee7ae0ef7525e2068120c6c8a3a08b2f58ca6f325ce"
GLSLANG_166_VERSION="16.6.0"
GLSLANG_166_URL="https://github.com/KhronosGroup/glslang/releases/download/${GLSLANG_166_VERSION}/glslang-${GLSLANG_166_VERSION}-windows-x86_64-release.zip"
GLSLANG_166_SHA256="82bf434e69b9bb4829de7e2b4bc2c5e7a7861e53d66cf75e5cc70f5f694a8d9b"
SHADERC_VERSION="2025.3"
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/win-64/latest"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS="$ROOT/.tools"
DOWNLOAD="$TOOLS/download"
GLSLANG_DIR="$TOOLS/glslang-${GLSLANG_VERSION}"
GLSLANG_166_DIR="$TOOLS/glslang-${GLSLANG_166_VERSION}"
ENV_DIR="$TOOLS/env"

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) ;;
    *)
        cat >&2 <<'EOF'
This script installs the Windows x86-64 builds that the pin names, because the
pinned glslang release assets are Windows zips. On Linux, take the same tools
from the distribution or the LunarG Vulkan SDK and put them on PATH (the checker
accepts several releases per compiler name and picks the one each artifact
records):
  glslangValidator 16.5.0 and 16.6.0, glslc (shaderc 2025.3), spirv-val
Then run the same two validation commands listed in the header.
EOF
        exit 1
        ;;
esac

win_path() {
    if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

mkdir -p "$DOWNLOAD"

# 1. glslang: the pinned release, whose binary is named `glslang` upstream and
#    `glslangValidator` by the legacy name the manifest and README use.
if [ ! -x "$GLSLANG_DIR/bin/glslang.exe" ] && [ ! -x "$GLSLANG_DIR/bin/glslang" ]; then
    zip="$DOWNLOAD/glslang-${GLSLANG_VERSION}-windows-x86_64-release.zip"
    if [ ! -f "$zip" ]; then
        echo "downloading glslang $GLSLANG_VERSION..."
        curl -fsSL -o "$zip" "$GLSLANG_URL"
    fi
    echo "$GLSLANG_SHA256  $zip" | sha256sum -c - >/dev/null || {
        echo "error: $zip does not match the published SHA-256 for glslang $GLSLANG_VERSION" >&2
        echo "       delete it and re-run to download it again" >&2
        exit 1
    }
    rm -rf "$GLSLANG_DIR"
    mkdir -p "$GLSLANG_DIR"
    unzip -q -o "$zip" -d "$GLSLANG_DIR"
    echo "installed glslang $GLSLANG_VERSION in $GLSLANG_DIR"
fi
if [ -x "$GLSLANG_DIR/bin/glslang.exe" ] && [ ! -e "$GLSLANG_DIR/bin/glslangValidator.exe" ]; then
    cp "$GLSLANG_DIR/bin/glslang.exe" "$GLSLANG_DIR/bin/glslangValidator.exe"
fi

# 1b. glslang 16.6.0: the second release the manifest pins. It is installed
#     beside 16.5.0, not instead of it; the checker selects per artifact.
if [ ! -x "$GLSLANG_166_DIR/bin/glslang.exe" ] && [ ! -x "$GLSLANG_166_DIR/bin/glslang" ]; then
    zip="$DOWNLOAD/glslang-${GLSLANG_166_VERSION}-windows-x86_64-release.zip"
    if [ ! -f "$zip" ]; then
        echo "downloading glslang $GLSLANG_166_VERSION..."
        curl -fsSL -o "$zip" "$GLSLANG_166_URL"
    fi
    echo "$GLSLANG_166_SHA256  $zip" | sha256sum -c - >/dev/null || {
        echo "error: $zip does not match the published SHA-256 for glslang $GLSLANG_166_VERSION" >&2
        echo "       delete it and re-run to download it again" >&2
        exit 1
    }
    rm -rf "$GLSLANG_166_DIR"
    mkdir -p "$GLSLANG_166_DIR"
    unzip -q -o "$zip" -d "$GLSLANG_166_DIR"
    echo "installed glslang $GLSLANG_166_VERSION in $GLSLANG_166_DIR"
fi
if [ -x "$GLSLANG_166_DIR/bin/glslang.exe" ] && [ ! -e "$GLSLANG_166_DIR/bin/glslangValidator.exe" ]; then
    cp "$GLSLANG_166_DIR/bin/glslang.exe" "$GLSLANG_166_DIR/bin/glslangValidator.exe"
fi

# 2. glslc (pinned shaderc) and spirv-val (SPIRV-Tools) from conda-forge, via a
#    micromamba unpacked into the same directory.
if [ ! -x "$ENV_DIR/Library/bin/glslc.exe" ] || [ ! -x "$ENV_DIR/Library/bin/spirv-val.exe" ]; then
    if [ ! -x "$DOWNLOAD/Library/bin/micromamba.exe" ]; then
        echo "downloading micromamba..."
        curl -fsSL -o "$DOWNLOAD/micromamba.tar.bz2" "$MICROMAMBA_URL"
        tar -xjf "$DOWNLOAD/micromamba.tar.bz2" -C "$DOWNLOAD"
    fi
    echo "creating the conda-forge environment (shaderc=$SHADERC_VERSION, spirv-tools)..."
    "$DOWNLOAD/Library/bin/micromamba.exe" create -y -p "$(win_path "$ENV_DIR")" \
        -c conda-forge --strict-channel-priority \
        "shaderc=$SHADERC_VERSION" spirv-tools
fi

# 3. The environment file, so callers get the pin rather than whatever is on PATH.
cat > "$TOOLS/env.sh" <<EOF
# Generated by validation/install_shader_toolchain.sh -- source this before
# compiling or provenance-checking shaders.
export PATH="$GLSLANG_DIR/bin:$ENV_DIR/Library/bin:\$PATH"
EOF

# 4. Verify the pin is what resolves, and say so out loud.
fail=0
glslang_banner="$("$GLSLANG_DIR/bin/glslangValidator.exe" --version 2>&1 | head -1 || true)"
case "$glslang_banner" in
    *"$GLSLANG_VERSION"*) echo "glslangValidator: $glslang_banner" ;;
    *) echo "error: expected glslang $GLSLANG_VERSION, got: $glslang_banner" >&2; fail=1 ;;
esac
glslang_166_banner="$("$GLSLANG_166_DIR/bin/glslangValidator.exe" --version 2>&1 | head -1 || true)"
case "$glslang_166_banner" in
    *"$GLSLANG_166_VERSION"*) echo "glslangValidator: $glslang_166_banner" ;;
    *) echo "error: expected glslang $GLSLANG_166_VERSION, got: $glslang_166_banner" >&2; fail=1 ;;
esac
if [ -x "$ENV_DIR/Library/bin/glslc.exe" ]; then
    echo "glslc:            $("$ENV_DIR/Library/bin/glslc.exe" --version 2>&1 | head -1)"
else
    echo "error: $ENV_DIR/Library/bin/glslc.exe is missing" >&2
    fail=1
fi
if [ -x "$ENV_DIR/Library/bin/spirv-val.exe" ]; then
    echo "spirv-val:        $("$ENV_DIR/Library/bin/spirv-val.exe" --version 2>&1 | head -1)"
else
    echo "error: $ENV_DIR/Library/bin/spirv-val.exe is missing" >&2
    fail=1
fi
[ "$fail" -eq 0 ] || exit 1

cat <<EOF

Sourced environment: .tools/env.sh

  source ../.tools/env.sh
  python validation/check_shader_provenance.py
  python validation/verify_training_shaders.py

The provenance checker compares every committed artifact with a rebuild from its
recorded recipe, compiled by the release that artifact records; the training-shader
release check additionally runs spirv-val over each rebuild and asserts the
recorded storage policy.
EOF

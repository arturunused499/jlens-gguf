#!/usr/bin/env bash
# Build jlens-server against a llama.cpp checkout.
#
#   ./build.sh                       # uses ../llama.cpp (the submodule)
#   LLAMA_DIR=/path/to/llama.cpp ./build.sh
#
# jlens-server depends ONLY on llama.cpp's PUBLIC API (llama.h, ggml*.h,
# libllama/libggml). The HTTP + JSON deps are vendored under native/vendor, so
# the build does not reach into llama.cpp's internals or its vendor/ layout —
# updating llama.cpp is just a rebuild. If llama.cpp is not built yet, this
# configures and builds the two libraries it needs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$SCRIPT_DIR/../llama.cpp}"
LLAMA_DIR="$(cd "$LLAMA_DIR" && pwd)"
BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_DIR/build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

if [ ! -f "$LLAMA_DIR/include/llama.h" ]; then
    echo "error: $LLAMA_DIR is not a llama.cpp checkout." >&2
    echo "       Run 'git submodule update --init' or set LLAMA_DIR." >&2
    exit 1
fi

# locate libllama (built shared or static)
find_lib() {
    for ext in so dylib a; do
        f=$(ls "$BUILD_DIR"/bin/libllama.$ext "$BUILD_DIR"/libllama.$ext 2>/dev/null | head -1 || true)
        [ -n "$f" ] && { echo "$f"; return; }
    done
}

if [ -z "$(find_lib)" ]; then
    echo "building llama.cpp in $BUILD_DIR (first time; a few minutes)..."
    cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "$BUILD_DIR" -j"$JOBS" --target llama
fi
LIB_DIR="$(dirname "$(find_lib)")"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2} -std=c++17 -pthread"
INC="-I$LLAMA_DIR/include -I$LLAMA_DIR/ggml/include -I$SCRIPT_DIR/vendor"
LIB="-L$LIB_DIR -lllama -lggml -lggml-base -Wl,-rpath,$LIB_DIR"

# vendored cpp-httplib ships split (.h decl + .cpp impl); compile the impl once
HTTPLIB_OBJ="$SCRIPT_DIR/httplib.o"
if [ ! -f "$HTTPLIB_OBJ" ] || [ "$SCRIPT_DIR/vendor/cpp-httplib/httplib.cpp" -nt "$HTTPLIB_OBJ" ]; then
    echo "compiling vendored cpp-httplib..."
    $CXX $CXXFLAGS -I"$SCRIPT_DIR/vendor/cpp-httplib" -c "$SCRIPT_DIR/vendor/cpp-httplib/httplib.cpp" -o "$HTTPLIB_OBJ"
fi

echo "compiling jlens-server (llama.cpp: $LIB_DIR)..."
$CXX $CXXFLAGS $INC -o "$SCRIPT_DIR/jlens-server" "$SCRIPT_DIR/jlens_server.cpp" "$HTTPLIB_OBJ" $LIB

echo "built: $SCRIPT_DIR/jlens-server"
"$SCRIPT_DIR/jlens-server" --help 2>&1 | head -2 || true

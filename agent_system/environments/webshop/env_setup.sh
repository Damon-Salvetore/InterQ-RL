#!/bin/bash
set -e
PREFIX=${WEBSHOP_PREFIX:-$HOME}

python3 -m virtualenv --version >/dev/null 2>&1 || pip install -q --user virtualenv
python3 -m virtualenv "$PREFIX/webshop_env"
"$PREFIX/webshop_env/bin/pip" install -q fastapi uvicorn gunicorn rank-bm25 pydantic pandas pyarrow tqdm spacy httpx pyserini pyjnius
"$PREFIX/webshop_env/bin/python" -m spacy download en_core_web_sm

curl -sL "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz" -o "$PREFIX/jdk21.tar.gz"
rm -rf "$PREFIX/jdk21"; mkdir -p "$PREFIX/jdk21"
tar xzf "$PREFIX/jdk21.tar.gz" -C "$PREFIX/jdk21" --strip-components=1

[ -d "$PREFIX/Agent-R1" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/0russwest0/Agent-R1 "$PREFIX/Agent-R1"

"$PREFIX/webshop_env/bin/python" "$(dirname "$0")/patch_env_server.py" "$PREFIX/Agent-R1"

echo "Done. Start the server with: WEBSHOP_PREFIX=$PREFIX WEBSHOP_DATA_DIR=<catalog dir> bash agent_system/environments/webshop/start_env_server.sh"

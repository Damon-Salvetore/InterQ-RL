#!/bin/bash
PREFIX=${WEBSHOP_PREFIX:-$HOME}
D=${WEBSHOP_DATA_DIR:?set WEBSHOP_DATA_DIR to the WebShop full-catalog directory}
if [[ ! -e "$D/products.sqlite" || ! -e "$D/lucene_index" ]]; then
    echo "FATAL: WebShop corpus incomplete at $D (need products.sqlite and lucene_index)" >&2
    exit 1
fi
D=$(cd "$D" && pwd)

export JAVA_HOME="$PREFIX/jdk21"
export PATH="$PREFIX/jdk21/bin:$PATH"
export JVM_PATH="$PREFIX/jdk21/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$PREFIX/jdk21/lib/server:$PREFIX/jdk21/lib"

export WEBSHOP_DATASET_MODE=full
export WEBSHOP_DATA_DIR="$D"
export WEBSHOP_INDEX_DIR="$D"
export WEBSHOP_SEARCH_TOP_K=50
export PYTHONPATH="$PREFIX/Agent-R1"

cd "$PREFIX/Agent-R1"
exec "$PREFIX/webshop_env/bin/gunicorn" -w 8 -k uvicorn.workers.UvicornWorker \
    recipes.webshop.env.server:app -b 127.0.0.1:4111 --timeout 300

#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

AGENT_R1 = pathlib.Path(sys.argv[1])
fc = AGENT_R1 / "recipes/webshop/env/full_catalog.py"

OLD_INIT = (
    '        uri = f"file:{self.db_path}?mode=ro"\n'
    "        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)\n"
    "        self._searcher = LuceneSearcher(str(self.lucene_index_dir))\n"
    "        self._product_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()\n"
)
NEW_INIT = (
    '        self._db_uri = f"file:{self.db_path}?mode=ro"\n'
    "        self._local = threading.local()\n"
    "        self._searcher = LuceneSearcher(str(self.lucene_index_dir))\n"
    "        self._product_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()\n"
    "        self._cache_lock = threading.Lock()\n"
    "\n"
    "    def _get_conn(self) -> sqlite3.Connection:\n"
    '        conn = getattr(self._local, "conn", None)\n'
    "        if conn is None:\n"
    "            conn = sqlite3.connect(self._db_uri, uri=True, check_same_thread=False)\n"
    "            self._local.conn = conn\n"
    "        return conn\n"
)

OLD_NUMPROD = (
    "    @property\n"
    "    def num_products(self) -> int:\n"
    "        row = self._conn.execute(\"SELECT value FROM meta WHERE key = 'num_products'\").fetchone()\n"
    "        if row:\n"
    "            return int(row[0])\n"
    '        row = self._conn.execute("SELECT COUNT(*) FROM products").fetchone()\n'
    "        return int(row[0])\n"
)
NEW_NUMPROD = (
    "    @property\n"
    "    def num_products(self) -> int:\n"
    "        row = self._get_conn().execute(\"SELECT value FROM meta WHERE key = 'num_products'\").fetchone()\n"
    "        if row:\n"
    "            return int(row[0])\n"
    '        row = self._get_conn().execute("SELECT COUNT(*) FROM products").fetchone()\n'
    "        return int(row[0])\n"
)

OLD_GETPROD = (
    "    def get_product(self, asin: str) -> dict[str, Any] | None:\n"
    "        asin = str(asin or \"\")\n"
    "        if asin in self._product_cache:\n"
    "            self._product_cache.move_to_end(asin)\n"
    "            return self._product_cache[asin]\n"
    '        row = self._conn.execute("SELECT product_json FROM products WHERE asin = ?", (asin,)).fetchone()\n'
    "        if not row:\n"
    "            return None\n"
    "        product = json.loads(row[0])\n"
    "        self._product_cache[asin] = product\n"
    "        if len(self._product_cache) > 20000:\n"
    "            self._product_cache.popitem(last=False)\n"
    "        return product\n"
)
NEW_GETPROD = (
    "    def get_product(self, asin: str) -> dict[str, Any] | None:\n"
    "        asin = str(asin or \"\")\n"
    "        with self._cache_lock:\n"
    "            if asin in self._product_cache:\n"
    "                self._product_cache.move_to_end(asin)\n"
    "                return self._product_cache[asin]\n"
    '        row = self._get_conn().execute("SELECT product_json FROM products WHERE asin = ?", (asin,)).fetchone()\n'
    "        if not row or not row[0]:\n"
    "            return None\n"
    "        try:\n"
    "            product = json.loads(row[0])\n"
    "        except (ValueError, TypeError):\n"
    "            return None\n"
    "        with self._cache_lock:\n"
    "            self._product_cache[asin] = product\n"
    "            if len(self._product_cache) > 20000:\n"
    "                self._product_cache.popitem(last=False)\n"
    "        return product\n"
)


def main() -> None:
    s = fc.read_text()
    orig = s
    if "self._local" in s or "_get_conn" in s:
        print("already patched, skip")
        return
    if "import threading" not in s:
        s = s.replace("import sqlite3\n", "import sqlite3\nimport threading\n", 1)
    s = s.replace(OLD_INIT, NEW_INIT, 1)
    s = s.replace(OLD_NUMPROD, NEW_NUMPROD, 1)
    s = s.replace(OLD_GETPROD, NEW_GETPROD, 1)
    assert s != orig, "patch did not apply; the source layout may have changed"
    assert "self._local = threading.local()" in s and "def _get_conn" in s, "__post_init__ not patched"
    assert "self._get_conn().execute" in s, "queries not patched"
    assert "except (ValueError, TypeError)" in s, "get_product not patched"
    assert "self._conn" not in s, "self._conn still referenced"
    fc.write_text(s)
    print(f"patched {fc}")


if __name__ == "__main__":
    main()

"""TEMPORARY: what the Connect popup says when Messages or WhatsApp can't be connected.

Throwaway test code, safe to delete: nothing imports this file, and it changes no app code.
It calls the connect endpoint directly (no server, no extra packages) against fake databases in a
temp folder, so it never reads your real Messages/WhatsApp data or writes to data/.

Run:  .venv/bin/python test_connect_failures.py
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

import imessage_extract
import store
import whatsapp_extract
from app import main
from app.main import ConnectRequest, connect_mcp

# The real table name each source checks for, so a fake db can get past check() and fail later.
TABLE = {"messages": "message", "whatsapp": "ZWAMESSAGE"}
MODULE = {"messages": imessage_extract, "whatsapp": whatsapp_extract}
SEEN = []  # (source, case, popup text), printed at the end


class ConnectFailures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        main.connections.clear()
        # Point the store at the temp folder so data/ is never touched.
        p = mock.patch.object(store, "DATA_DIR", self.tmp / "data")
        p.start()
        self.addCleanup(p.stop)

    def popup(self, source, case, db_path):
        """Connect with this path and return the error the popup would show (the HTTP detail)."""
        with self.assertRaises(HTTPException) as ctx:
            connect_mcp(source, ConnectRequest(fields={"db_path": str(db_path)}))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertNotIn(source, main.connections, "a failed connect must not mark the source connected")
        SEEN.append((source, case, ctx.exception.detail))
        return ctx.exception.detail

    def each_source(self):
        for source in ("messages", "whatsapp"):
            with self.subTest(source=source):
                yield source

    def test_no_database_at_path(self):
        for s in self.each_source():
            msg = self.popup(s, "No file at the path", self.tmp / "missing.db")
            self.assertIn("No database found", msg)

    def test_blank_path_and_app_not_installed(self):
        for s in self.each_source():
            with mock.patch.object(MODULE[s], "DB_PATH", self.tmp / "not-installed.db"):
                msg = self.popup(s, "Field left blank, app not installed", "")
            self.assertIn("No database found", msg)

    def test_macos_blocks_access(self):
        for s in self.each_source():
            db = self.tmp / f"{s}-locked.db"
            sqlite3.connect(db).close()
            db.chmod(0)  # unreadable, like a file behind Full Disk Access
            self.addCleanup(db.chmod, 0o600)
            msg = self.popup(s, "No permission (Full Disk Access)", db)
            self.assertIn("Full Disk Access", msg)

    def test_not_a_database(self):
        for s in self.each_source():
            f = self.tmp / f"{s}-notes.txt"
            f.write_text("just some text, not sqlite\n" * 50)
            msg = self.popup(s, "File isn't a database", f)
            self.assertIn("doesn't look like the right database", msg)

    def test_wrong_database(self):
        for s in self.each_source():
            db = self.tmp / f"{s}-other.db"
            with sqlite3.connect(db) as c:
                c.execute("CREATE TABLE something_else (x)")
            msg = self.popup(s, "Some other app's database", db)
            self.assertIn("doesn't look like the right database", msg)

    def test_extraction_fails(self):
        # The table check() looks for exists, but without the columns the extractor queries.
        for s in self.each_source():
            db = self.tmp / f"{s}-broken.db"
            with sqlite3.connect(db) as c:
                c.execute(f"CREATE TABLE {TABLE[s]} (x)")
            msg = self.popup(s, "Opens, but extraction fails", db)
            self.assertIn("couldn't read its messages", msg)
            self.assertNotIn("no such column", msg, "the raw SQLite error belongs in the server log, not the popup")


def tearDownModule():
    print("\nWhat the Connect popup shows:\n")
    for source, case, text in SEEN:
        print(f"[{main.MCPS_BY_ID[source]['name']}] {case}\n    {text}\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)


import sqlite3

class DBLogger:
    def __init__(self):
        self.conn = sqlite3.connect("drift_logs.db")
        self.create_table()

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT
        )
        """)

    def insert_event(self, event):
        self.conn.execute("INSERT INTO drift_events(event) VALUES (?)", (event,))
        self.conn.commit()

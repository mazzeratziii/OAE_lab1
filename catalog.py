import sqlite3
from contextlib import closing
from pathlib import Path

DB = Path(__file__).with_name("catalog.sqlite3")
DATA = {
    "laptop": [("Acer Aspire", 45000), ("Lenovo IdeaPad", 59000),
               ("ASUS VivoBook", 65000), ("HP Pavilion", 79000), ("MacBook Air", 110000)],
    "smartphone": [("Redmi Note", 18000), ("Galaxy A35", 29000),
                   ("Pixel 8a", 48000), ("iPhone 15", 75000), ("Galaxy S24 Ultra", 115000)],
    "tablet": [("Lenovo Tab", 15000), ("Redmi Pad", 21000),
               ("Galaxy Tab A9", 27000), ("iPad Air", 68000), ("Galaxy Tab S9", 82000)],
}


def initialize(path=DB):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            category TEXT NOT NULL CHECK(category IN ('laptop', 'smartphone', 'tablet')),
            price INTEGER NOT NULL CHECK(price >= 0), UNIQUE(category, name))""")
        db.executemany(
            "INSERT INTO products(name, category, price) VALUES (?, ?, ?) "
            "ON CONFLICT(category, name) DO NOTHING",
            [(name, category, price) for category, items in DATA.items()
             for name, price in items],
        )


def search(query, path=DB):
    if query.category == "other":
        return []
    sql = "SELECT id, name, category, price FROM products WHERE category = ?"
    values = [query.category]
    if query.max_price is not None:
        sql += " AND price <= ?"
        values.append(query.max_price)
    direction = {"price_asc": "ASC", "price_desc": "DESC"}[query.sort_by]
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f"{sql} ORDER BY price {direction}, id", values)]


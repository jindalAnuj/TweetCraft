# db_module.py
import pickledb

class DBHandler:
    def __init__(self, db_path="test.db"):
        create_if_not_exists = True
        self.db = pickledb.load(db_path, create_if_not_exists)
    
    def set(self, key, value):
        self.db.set(key.encode(), value.encode())

    def get(self, key):
        value = self.db.get(key.encode())
        return value.decode() if value else None
    
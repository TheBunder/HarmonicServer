import logging
import sqlite3
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib, uuid


class DBHelper:

    TABLE_NAME = "users"
    COLUMN_USERNAME = "username"
    COLUMN_PASSWORD = "password"
    COLUMN_SALT = "salt"

    ITERATION_COUNT = 1024  # Adjust as needed
    KEY_LENGTH = 256  # Adjust as needed

    def __init__(self, db_name="Login.db"):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """Creates the 'users' table if it doesn't exist."""
        query = f"CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} ({self.COLUMN_USERNAME} TEXT PRIMARY KEY, {self.COLUMN_SALT} BLOB, {self.COLUMN_PASSWORD} TEXT)"
        self.cursor.execute(query)
        self.connection.commit()

    def insert_data(self, username, password):
        """Inserts a new user with hashed password and salt."""
        salt = uuid.uuid4().bytes  # Generate random salt
        hashed_password = hashlib.sha512(password.encode() + salt).hexdigest()

        if hashed_password:
            values = {
                self.COLUMN_USERNAME: username,
                self.COLUMN_PASSWORD: hashed_password,
                self.COLUMN_SALT: salt,
            }
            try:
                self.cursor.execute(
                    f"INSERT INTO {self.TABLE_NAME} ({','.join(values.keys())}) VALUES ({','.join(['?'] * len(values))})",
                    tuple(values.values()),
                )
                self.connection.commit()
                return True
            except Exception as e:
                logging.info(f"Error inserting data: {e}")
                return False

    def check_username(self, username):
        """Checks if a username exists in the database."""
        query = f"SELECT * FROM {self.TABLE_NAME} WHERE {self.COLUMN_USERNAME} = ?"
        self.cursor.execute(query, (username,))
        return self.cursor.fetchone() is not None  # Check if any rows returned

    def check_username_password(self, username, password):
        """Checks if the username and password match an existing user."""
        query = f"SELECT {self.COLUMN_SALT} FROM {self.TABLE_NAME} WHERE {self.COLUMN_USERNAME} = ?"
        self.cursor.execute(query, (username,))
        salt = self.cursor.fetchone()

        if salt:
            salt = salt[0]  # Extract salt from retrieved row
            hashed_password = hashlib.sha512(password.encode() + salt).hexdigest()
            if hashed_password:
                query = f"SELECT * FROM {self.TABLE_NAME} WHERE {self.COLUMN_USERNAME} = ? AND {self.COLUMN_PASSWORD} = ?"
                self.cursor.execute(query, (username, hashed_password))
                return self.cursor.fetchone() is not None
        return False

    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

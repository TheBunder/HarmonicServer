import hashlib
import logging
import sqlite3
import uuid


class DBHelper:
    TABLE_NAME_USERS = "users"
    COLUMN_USERNAME = "username"
    COLUMN_PASSWORD = "password"
    COLUMN_SALT = "salt"

    TABLE_NAME_SOUNDS = "sounds"
    COLUMN_USERNAME = "username"
    COLUMN_SOUND_NAME = "sound_name"
    COLUMN_FILE_NAME = "file_name"

    def __init__(self, db_name="Login.db"):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """Creates the 'users' table if it doesn't exist."""
        query = f"CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, salt BLOB, password TEXT)"
        self.cursor.execute(query)
        self.connection.commit()
        """Creates the 'sounds' table if it doesn't exist."""
        query = f"CREATE TABLE IF NOT EXISTS sounds (sound_name TEXT,file_name TEXT,userid int,FOREIGN KEY(userid) REFERENCES users(username))"
        self.cursor.execute(query)
        self.connection.commit()

    def insert_data_to_users_table(self, username, password):
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
                    f"INSERT INTO users ({','.join(values.keys())}) VALUES ({','.join(['?'] * len(values))})",
                    tuple(values.values()),
                )
                self.connection.commit()
                return True
            except Exception as e:
                logging.info(f"Error inserting data: {e}")
                return False

    def insert_data_to_sounds_table(self, username, sound_name, file_name):
        try:
            self.cursor.executemany(
                f"INSERT INTO sounds (userid, sound_name, file_name) VALUES (?,?,?)",
                [(username, sound_name, file_name)],
            )
            self.connection.commit()
            return True
        except Exception as e:
            logging.info(f"Error inserting data: {e}")
            return False

    def check_username(self, username):
        """Checks if a username exists in the database."""
        query = f"SELECT * FROM users WHERE username = ?"
        self.cursor.execute(query, (username,))
        return self.cursor.fetchone() is not None  # Check if any rows returned

    def check_username_password(self, username, password):
        """Checks if the username and password match an existing user."""
        query = f"SELECT salt FROM users WHERE username = ?"
        self.cursor.execute(query, (username,))
        salt = self.cursor.fetchone()

        if salt:
            salt = salt[0]  # Extract salt from retrieved row
            hashed_password = hashlib.sha512(password.encode() + salt).hexdigest()
            if hashed_password:
                query = f"SELECT * FROM users WHERE username = ? AND password = ?"
                self.cursor.execute(query, (username, hashed_password))
                return self.cursor.fetchone() is not None
        return False

    def check_user_sounds(self, username):
        query = f"SELECT sounds.sound_name FROM sounds JOIN users ON sounds.userid = users.username WHERE users.username = ?;"
        # Using parameterized query to avoid SQL injection
        self.cursor.execute(query, (username,))
        # Fetch all rows
        rows = self.cursor.fetchall()
        # Extract sound names from the fetched rows
        sound_names = [row[0] for row in rows]
        return sound_names

    def get_file_name_from_sound(self, sound_name, username):
        """Get the file name associated with a sound name."""
        query = "SELECT file_name FROM sounds WHERE sound_name = ? AND userid= ?"
        # Using parameterized query to avoid SQL injection
        self.cursor.execute(query, (sound_name, username))
        result = self.cursor.fetchone()
        if result:
            return result[0]  # Returning the file name
        else:
            return None  # Return None if no matching sound found

    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

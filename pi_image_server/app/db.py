import psycopg2

conn = psycopg2.connect(host="localhost",
                        dbname="pi_image_db",
                        user="jacobfishel",
                        password="Biker129",
                        port=5432)

cur = conn.cursor()


cur.execute("""CREATE TABLE IF NOT EXISTS person (
            id INT PRIMARY KEY,
            name VARCHAR(255),
            age INT,
            gender char
            );
""")

cur.execute("""CREATE TABLE IF NOT EXISTS image (
            id INT PRIMARY KEY,
            file_location TEXT,
            tags TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
""")


conn.commit()

cur.close()
conn.close()
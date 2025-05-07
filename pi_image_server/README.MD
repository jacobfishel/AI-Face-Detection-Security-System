notes:
to run and inspect the db:
    psql -U jacobfishel -d postgres

SQL CHEATSHEET:

\dt : List all tables in current database

📁 Database Commands
Command	What It Does
CREATE DATABASE dbname;	Creates a new database
DROP DATABASE dbname;	Deletes a database
\l (PostgreSQL)	Lists all databases
\c dbname (PostgreSQL)	Connects to a database

📋 Table Commands
Command	What It Does
CREATE TABLE table (...);	Creates a new table
DROP TABLE table;	Deletes a table
ALTER TABLE table ADD column ...;	Adds a new column
ALTER TABLE table DROP COLUMN ...;	Deletes a column
TRUNCATE TABLE table;	Deletes all rows (but keeps the table)
\dt (PostgreSQL)	Lists tables
\d table (PostgreSQL)	Describes table structure

📥 Data Insertion
Command	What It Does
INSERT INTO table (col1, col2) VALUES (val1, val2);	Adds a new row

📤 Data Selection
Command	What It Does
SELECT * FROM table;	Selects all columns and rows
SELECT col1, col2 FROM table;	Selects specific columns
SELECT * FROM table WHERE condition;	Filters rows
`ORDER BY col ASC	DESC;`
LIMIT n;	Limits number of rows returned

✏️ Data Updating
Command	What It Does
UPDATE table SET col='val' WHERE condition;	Updates rows

❌ Data Deletion
Command	What It Does
DELETE FROM table WHERE condition;	Deletes specific rows
DELETE FROM table;	Deletes all rows (with caution)

🧠 Schema Introspection
Command	What It Does
\dt	Lists tables (PostgreSQL)
\d tablename	Describes a table
SELECT * FROM information_schema.tables;	Lists all tables
SELECT column_name FROM information_schema.columns WHERE table_name='your_table';	Lists columns

🔐 Permissions & Users (PostgreSQL specific)
Command	What It Does
CREATE USER username WITH PASSWORD 'password';	Creates a user
GRANT ALL PRIVILEGES ON DATABASE db TO user;	Grants privileges
GRANT SELECT, INSERT ON table TO user;	Grants specific permissions
\du	Lists users/roles
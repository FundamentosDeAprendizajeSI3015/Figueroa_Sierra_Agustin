import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import os
import getpass

# 1. Database Configuration
# Change 'root' and 'localhost' if your setup is different
DB_USER = input("MySQL Username (default 'root'): ") or "root"
DB_PASS = getpass.getpass("MySQL Password: ")
DB_HOST = "127.0.0.1"
DB_NAME = "weatherchart"
TABLE_NAME = "train_dataset"

# 2. File Configuration
FILE_PATH = r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\train_dataset.csv"

if not os.path.exists(FILE_PATH):
    print(f"Error: File not found at {FILE_PATH}")
    exit(1)

# 3. Create Connection
try:
    connection_str = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(connection_str)
    print("Connecting to database...")
    # Test connection
    with engine.connect() as conn:
        print("Successfully connected!")
except Exception as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)

# 4. Import in Chunks
CHUNK_SIZE = 50000  # Adjust based on memory
TOTAL_ROWS = 18401182

print(f"Starting import of {TOTAL_ROWS:,} rows...")

# First chunk with 'replace' to ensure schema, then 'append'
first_chunk = True

# Progress bar
pbar = tqdm(total=TOTAL_ROWS, desc="Importing data", unit="rows")

try:
    for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE):
        # Convert date column to ensure SQL compatibility
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'])
        
        mode = 'replace' if first_chunk else 'append'
        chunk.to_sql(TABLE_NAME, engine, if_exists=mode, index=False)
        
        pbar.update(len(chunk))
        first_chunk = False

    pbar.close()
    print("\n✅ Import finished successfully!")
    
except Exception as e:
    pbar.close()
    print(f"\n❌ Error during import: {e}")

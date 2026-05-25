import os
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING"
)
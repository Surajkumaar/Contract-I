from motor.motor_asyncio import AsyncIOMotorClient
import gridfs

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "contracts_db"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
fs = gridfs.GridFS(db.delegate)

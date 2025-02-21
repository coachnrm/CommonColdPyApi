from fastapi import FastAPI, Depends, Query
import pandas as pd
import aiomysql
import asyncio
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = FastAPI()

# Database Configuration
DB_CONFIG = {
    "host": "mariadb",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "db": "test",
    "autocommit": True
}

# Global variable to store the trained model
model = None

# Function to manage database connection
async def get_db():
    pool = await aiomysql.create_pool(**DB_CONFIG)
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            yield cur
    pool.close()
    await pool.wait_closed()

# Function to fetch data asynchronously and train the model
async def fetch_and_train_model():
    global model
    try:
        async with aiomysql.create_pool(**DB_CONFIG) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute("SELECT * FROM ColdSyndromeInsert")
                    result = await cur.fetchall()

        # Convert result to DataFrame
        df = pd.DataFrame(result)

        if df.empty:
            print("No data found in the database. Skipping model training.")
            return None

        # Ensure correct data format
        X = df[['Head', 'Nose', 'Neck', 'Fever']]
        y = df['CommonCold']

        # Train a RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X, y)

        print("Model training completed successfully.")

    except Exception as e:
        print(f"Error fetching and training model: {e}")
        model = None

# FastAPI startup event
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(fetch_and_train_model())

# API endpoint to predict cold syndrome
@app.get("/predict/")
async def predict(
    head: int = Query(..., description="ปวดศีรษะ: 0 (ไม่มีอาการ), 1 (มีอาการ)"),
    nose: int = Query(..., description="น้ำมูกไหล: 0 (ไม่มีอาการ), 1 (มีอาการ)"),
    neck: int = Query(..., description="เจ็บคอ: 0 (ไม่มีอาการ), 1 (มีอาการ)"),
    fever: int = Query(..., description="ไข้: 0 (ไม่มีอาการ), 1 (มีอาการ)")
):
    try:
        if model is None:
            return {"error": "Model is not available yet. Please try again later."}

        # Prepare input for prediction
        input_data = pd.DataFrame([[head, nose, neck, fever]], columns=['Head', 'Nose', 'Neck', 'Fever'])

        # Make prediction
        result = model.predict(input_data)[0]

        # Convert result to readable output
        diagnosis = "เป็นไข้หวัด" if result == 1 else "ไม่เป็นไข้หวัด"

        return {"prediction": diagnosis}

    except Exception as e:
        return {"error": str(e)}

@app.get("/get-data/")
async def get_data():
    """
    Fetches the latest 10 records from the ColdSyndromeInsert table.
    """
    try:
        # Create a database connection
        pool = await aiomysql.create_pool(**DB_CONFIG)
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM ColdSyndromeInsert LIMIT 10")
                result = await cur.fetchall()

        pool.close()
        await pool.wait_closed()

        if not result:
            return {"message": "No records found."}

        return {"data": result}

    except Exception as e:
        return {"error": str(e)}


# Run FastAPI app (for local development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



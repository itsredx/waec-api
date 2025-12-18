import os
import io
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Global State ---
models = {}
DATA_FILE = "waec_data.csv"
MODEL_FILE = "waec_model.pkl"
GENDER_ENCODER_FILE = "gender_encoder.pkl"
SCHOOL_ENCODER_FILE = "school_encoder.pkl"

# --- Models ---
class PredictInput(BaseModel):
    year: int
    gender: str  # Male, Female
    school_type: str # Public, Private
    total_sat: int

class ChatInput(BaseModel):
    message: str

class FeedbackInput(BaseModel):
    year: int
    state: str
    zone: str
    scholl_type: str  # Intentionally misspelled to match dataset
    gender: str
    total_sat: int
    pass_eng: int
    pass_math: int
    pass_eng_math: int
    pass_percent: float

# --- Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Models and Encoders
    try:
        print("Loading models...")
        models["model"] = joblib.load(MODEL_FILE)
        models["gender_encoder"] = joblib.load(GENDER_ENCODER_FILE)
        models["school_encoder"] = joblib.load(SCHOOL_ENCODER_FILE)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # We don't raise here to allow app to start, but endpoints will fail
    
    # Check Data File
    if not os.path.exists(DATA_FILE):
        print(f"Warning: {DATA_FILE} not found. Some features may not work.")
    else:
        print(f"Found {DATA_FILE}.")

    yield
    # Cleanup if needed
    models.clear()

# --- App Setup ---
app = FastAPI(title="WAEC Analytics API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Clients ---
# Initialize Groq client
# Ensure GROQ_API_KEY is set in your environment variables
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("Warning: GROQ_API_KEY not found. Chat endpoint will fail.")
    groq_client = None
else:
    groq_client = Groq(api_key=api_key)

# --- Helpers ---
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    return pd.read_csv(DATA_FILE)

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "WAEC Analytics API is running"}

@app.post("/predict")
def predict_pass_rate(input_data: PredictInput):
    if "model" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode inputs
        try:
            # Check if gender exists in encoder classes
            if input_data.gender not in models["gender_encoder"].classes_:
                 raise HTTPException(status_code=400, detail=f"Invalid gender. Must be one of {list(models['gender_encoder'].classes_)}")
            
            gender_encoded = models["gender_encoder"].transform([input_data.gender])[0]
            
            # Check if school_type exists (handling the 'scholl' typo from dataset if retained in encoder)
            # The user code has 'scholl_type' in dataset, let's assume encoder expects what was in dataset.
            # We map input "school_type" -> dataset "scholl_type"
            
            # Only transforming if valid
            if input_data.school_type not in models["school_encoder"].classes_:
                 raise HTTPException(status_code=400, detail=f"Invalid school_type. Must be one of {list(models['school_encoder'].classes_)}")

            school_encoded = models["school_encoder"].transform([input_data.school_type])[0]
            
        except AttributeError:
             # Fallback if classes_ isn't available or other sklearn breakdown
             raise HTTPException(status_code=500, detail="Encoder error")

        # Create DataFrame for model
        # Feature order must match training: ['year', 'gender_encoded', 'scholl_type_encoded', 'total_sat']
        features = pd.DataFrame({
            'year': [input_data.year],
            'gender_encoded': [gender_encoded],
            'scholl_type_encoded': [school_encoded],
            'total_sat': [input_data.total_sat]
        })

        prediction = models["model"].predict(features)[0]
        return {"predicted_pass_rate": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_data(input_data: ChatInput):
    df = load_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not available for analysis")

    # Convert data to string (truncate if necessary, though dataset is likely small)
    # Using to_markdown() or to_csv()
    data_str = df.to_csv(index=False)
    
    # Truncate to approx 15k chars to be safe for context window if it grows
    if len(data_str) > 20000:
        data_str = data_str[:20000] + "\n...(truncated)"

    system_prompt = (
        "You are an expert Data Analyst for WAEC exam data.\n"
        f"Here is the raw dataset you have access to:\n{data_str}\n\n"
        "Answer the user's question based strictly on this data. Be concise and insightful."
    )

    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq Client not initialized. Missing API Key.")

    async def generate():
        try:
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_data.message},
                ],
                temperature=1,
                max_completion_tokens=8192,
                top_p=1,
                reasoning_effort="medium",
                stream=True,
                stop=None
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            # In a stream, we can't easily raise HTTP exception once started, 
            # but usually it fails immediately.
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/feedback")
def submit_feedback(data: FeedbackInput):
    # Prepare row
    new_row = {
        'year': data.year,
        'state': data.state,
        'zone': data.zone,
        'scholl_type': data.scholl_type,
        'gender': data.gender,
        'total_sat': data.total_sat,
        'pass_eng': data.pass_eng,
        'pass_math': data.pass_math,
        'pass_eng_math': data.pass_eng_math,
        'pass_%': data.pass_percent
    }
    
    # Calculate derived if missing? The user prompt implies appending.
    # We append to CSV.
    try:
        df = load_data()
        new_df = pd.DataFrame([new_row])
        combined_df = pd.concat([df, new_df], ignore_index=True)
        combined_df.to_csv(DATA_FILE, index=False)
        
        # Trigger Retrain Stub
        # In a real app, this would be a background task
        # retrain_model(combined_df) 
        
        return {"status": "success", "message": "Feedback received and data saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")

@app.get("/dashboard-data")
def get_dashboard_data():
    df = load_data()
    if df.empty:
        return {"error": "No data available"}
        
    try:
        # 1. Gender Trend (Yearly Pass % by Gender)
        # Assuming we aggregate mean pass_percent for simplicity
        gender_trend_df = df.groupby(['year', 'gender'])['pass_%'].mean().unstack().reset_index()
        # Rename for clarity: year, Female, Male
        gender_trend = gender_trend_df.to_dict(orient='records')
        
        # 2. School Performance (Public vs Private)
        school_perf_df = df.groupby(['year', 'scholl_type'])['pass_%'].mean().unstack().reset_index()
        school_performance = school_perf_df.to_dict(orient='records')
        
        # 3. Subject Performance (Aggregate sums per year)
        # Summing the counts
        subject_cols = ['pass_eng', 'pass_math', 'pass_eng_math']
        subject_perf_df = df.groupby('year')[subject_cols].sum().reset_index()
        subject_perf_df = subject_perf_df.rename(columns={
            'pass_eng': 'English',
            'pass_math': 'Math',
            'pass_eng_math': 'Both'
        })
        subject_performance = subject_perf_df.to_dict(orient='records')

        return {
            "gender_trend": gender_trend,
            "school_performance": school_performance,
            "subject_performance": subject_performance
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error processing stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

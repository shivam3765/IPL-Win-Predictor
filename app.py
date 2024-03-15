from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import pandas as pd


# ====================== create object of the FastAPI class ========================
app = FastAPI()


# ================ Mount the static files ===================
app.mount("/static", StaticFiles(directory="templates", html=True))
templates = Jinja2Templates(directory="templates")


# ==================== Load model and final data =================
pipe = pickle.load(open('pipe.pkl', 'rb'))
final_df = pickle.load(open('final_df.pkl', 'rb'))


# ===================== Create route for render html file =======================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cities = sorted(final_df['city'].unique())
    teams = sorted(final_df['batting_team'].unique())

    cities.insert(0, 'Select City')
    teams.insert(0, 'Select Team')
    return templates.TemplateResponse('home.html', 
        {'request': request, 'cities': cities, 'teams': teams})



# ==================== Create route for prediction and redirect to the html file ==================
@app.post("/predict")
async def predict(request: Request,
                  batting_team: str = Form(...),
                  bowling_team: str = Form(...),
                  selected_city: str = Form(...),
                  target: int = Form(...),
                  score: int = Form(...),
                  overs: int = Form(...),
                  wickets: int = Form(...)):
    
    try:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left],
                'wickets_left': [wickets], 'total_runs_x': [target], 'current_rr': [crr],
                'required_rr': [rrr]})
        
        prediction = pipe.predict_proba(input_df)

        loss = prediction[0][0]
        win = prediction[0][1]

        return {"win": str(round(win * 100)), "loss": str(round(loss * 100)), "batting": batting_team, "bowling": bowling_team}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

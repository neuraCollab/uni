from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import json

templates = Jinja2Templates(directory="templates/json")
router = APIRouter()

current_dir = os.path.dirname(__file__)
DATA_FILE = os.path.join(current_dir, '../../bot/tokens.json')

def read_data():
    try:
        with open(DATA_FILE, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data

def write_data(data):
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file)

@router.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    data = read_data()
    return templates.TemplateResponse("index.html", {"request": request, "tokens": data})

@router.post("/add_token/")
async def add_token(request: Request, new_token: str = Form(...)):
    data = read_data()
    if new_token not in data:
        data.append(new_token)
        write_data(data)
        # После добавления токена перенаправляем обратно на главную страницу
        return RedirectResponse(url='../', status_code=303)
    # Можно добавить обработку ошибки, если токен уже существует
    return templates.TemplateResponse("index.html", {"request": request, "tokens": data, "error": "Token already exists."})

@router.get("/delete_token/")
async def delete_token(token: str):
    data = read_data()
    if token in data:
        data.remove(token)
        write_data(data)
        return {"status": "success", "message": f"Token '{token}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Token not found")

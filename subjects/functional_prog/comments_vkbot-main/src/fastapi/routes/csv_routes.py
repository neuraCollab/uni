from fastapi import APIRouter, HTTPException
from fastapi import  Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import shutil
import csv
import os

# Импортируйте необходимые функции и классы
templates = Jinja2Templates(directory="templates/csv")

# Получение текущей директории файла
current_dir = os.path.dirname(__file__)

# Директория для хранения и чтения CSV-файлов
csv_dir = os.path.join(current_dir, '../../csv')
os.makedirs(csv_dir, exist_ok=True)

router = APIRouter()

def get_csv_files():
    """Получает список CSV-файлов в директории."""
    return [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    csv_files = get_csv_files()
    # print(csv_files)
    return templates.TemplateResponse("index.html", {"request": request, "csv_files": csv_files})

@router.post("/create")
async def create_csv(file_name: str = Form(...)):
    if file_name.endswith('.csv'):
        with open(os.path.join(csv_dir, file_name), 'w', newline='') as csvfile:
            csv.writer(csvfile).writerow(["Column1", "Column2"])  # Заголовки по умолчанию
    return RedirectResponse(url="/csv", status_code=303)

@router.get("/edit/{file_name}", response_class=HTMLResponse)
async def edit_csv(request: Request, file_name: str):
    # Ensure the file_name ends with '.csv'
    if not file_name.endswith('.csv'):
        # Handle the error appropriately
        raise HTTPException(status_code=404, detail="File not found or invalid format")

    rows = []
    print(rows)
    try:
        with open(os.path.join(csv_dir, file_name), 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                rows.append(row)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    # return RedirectResponse(url=f"/csv/edit/{file_name}", status_code=303)
    return templates.TemplateResponse("edit.html", {"request": request, "file_name": file_name, "rows": rows})

# @router.post("/update/{file_name}")
# async def update_csv(file_name: str, request: Request):
#     form_data = await request.form()
#     updated_rows = form_data.getlist('row')
#     with open(os.path.join(csv_dir, file_name), 'w', newline='') as csvfile:
#         csv.writer(csvfile).writerows([row.split(',') for row in updated_rows])
#     return RedirectResponse(url=f"/csv/edit/{file_name}", status_code=303)
    # return {"message": "CSV updated", "file_name": file_name}

@router.post("/update/{file_name}")
async def update_csv(file_name: str, request: Request):
    form_data = await request.form()
    updated_rows = form_data.getlist('row')  # Get list of rows

    with open(os.path.join(csv_dir, file_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in updated_rows:
            writer.writerow(row.split(','))  # Split each row string into a list of values

    return RedirectResponse(url=f"/csv/edit/{file_name}", status_code=303)


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...), file_name: str = Form(None)):
    # Use the provided file name or fall back to the original file name
    filename_to_use = file_name or file.filename
    file_location = os.path.join(csv_dir, filename_to_use)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"info": f"file '{filename_to_use}' saved at '{file_location}'"}


@router.get("/delete/{file_name}")
async def delete_file(file_name: str):
    file_path = os.path.join(csv_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"{file_name} deleted."}
    else:
        raise HTTPException(status_code=404, detail="File not found")
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from routes import csv_routes, token_routes

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.include_router(csv_routes.router, prefix="/csv")
app.include_router(token_routes.router, prefix="/token")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    routes = [route for route in app.routes]
    return templates.TemplateResponse("index.html", {"request": request, "routes": routes})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)

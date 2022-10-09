import uvicorn
from fastapi import FastAPI
from routers import images

app = FastAPI()
app.include_router(images.router)


@app.get("/")
async def root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

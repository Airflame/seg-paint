import uvicorn
from fastapi import FastAPI
from routers import thresholding, segmentation

app = FastAPI()
app.include_router(thresholding.router)
app.include_router(segmentation.router)


@app.get("/")
async def root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

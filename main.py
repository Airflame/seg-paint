import uvicorn
from fastapi import FastAPI
from routers import thresholding_router, segmentation_router, metrics_router

app = FastAPI()
app.include_router(thresholding_router.router)
app.include_router(segmentation_router.router)
app.include_router(metrics_router.router)


@app.get("/")
async def root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

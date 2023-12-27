from fastapi import FastAPI

from LLM_arch.minGPT.mingpt.servable import GPTServable
from ..services.inference import run_inference


app = FastAPI(
    title="LLM Learning API",
    description="API for LLM Learning",
    version="0.1.0",
)


@app.get("/infer")
async def run_inference():
    return await run_inference()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



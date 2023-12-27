import uvicorn


def main():
    uvicorn.run(
        "sandbox.api.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        reload=True,
        # reload_dirs=["sandbox"],
    )


if __name__ == "__main__":
    main()

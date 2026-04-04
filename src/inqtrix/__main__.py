""" Entry point: python -m inqtrix """

import uvicorn

from inqtrix.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5100, workers=1, timeout_keep_alive=300)

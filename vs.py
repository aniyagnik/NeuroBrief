"""NeuroBrief entry point — run with: python vs.py"""

import os

from neurobrief.routes import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    use_reloader = os.getenv("FLASK_USE_RELOADER", "0") == "1"
    from neurobrief.config import logger

    logger.info("Listening on http://0.0.0.0:%d (debug=%s)", port, debug)
    app.run(
        debug=debug,
        host="0.0.0.0",
        port=port,
        threaded=True,
        use_reloader=use_reloader,
    )

from pathlib import Path

from flask import Flask, redirect, request

from app.db import init_db
from app.routes import main_bp


def create_app(redirect_to_https: bool = False) -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)

    app.config.from_mapping(
        DATABASE=str(instance_path / "vesta.db"),
    )

    # If this is the HTTP redirect instance, redirect all traffic to HTTPS
    if redirect_to_https:
        @app.before_request
        def redirect_to_https_handler():
            # Get the host from the request
            host = request.host.split(':')[0]  # Remove port if present
            return redirect(f'https://{host}{request.path}', code=301)
    
    init_db(app)
    app.register_blueprint(main_bp)

    return app

from pathlib import Path

from flask import Flask

from app.db import init_db
from app.routes import main_bp


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)

    app.config.from_mapping(
        DATABASE=str(instance_path / "vesta.db"),
    )

    init_db(app)
    app.register_blueprint(main_bp)

    return app

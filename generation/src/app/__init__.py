from flask import Flask
from .routes import main_bp

def create_app(config_class="config.Config"):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Register our simple routes
    app.register_blueprint(main_bp)
    
    return app
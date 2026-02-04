from src.app import create_app
import os

config_name = os.environ.get('APP_SETTINGS', 'DevConfig')

app = create_app(f"config.{config_name}")

if __name__ == "__main__":
    app.run()
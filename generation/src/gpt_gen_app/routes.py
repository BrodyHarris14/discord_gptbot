from flask import Blueprint

main_bp = Blueprint('main', __name__)

@main_bp.route('/generate', methods=['GET', 'POST'])
def generate():
    # Pure stateless response
    return "HELLO", 200
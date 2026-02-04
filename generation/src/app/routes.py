from src.gen.generate_samples import generate_text
from flask import Blueprint, request

main_bp = Blueprint('main', __name__)

@main_bp.route('/generate', methods=['GET', 'POST'])
def generate():
    set = request.args.get('set')
    prefix = request.args.get('prefix')

    if set is None or prefix is None:
       return "Missing 'set' or 'prefix' parameter", 400

    try:
     text = generate_text(set, prefix)
    except Exception as e:
        return str(e), 500

    return text, 200
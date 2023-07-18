from flask import Blueprint
from controllers.ml_controller import ML_router

router = Blueprint("router", __name__)

router.register_blueprint(ML_router)

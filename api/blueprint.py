from flask import Blueprint, jsonify
from flask import request

from api.service import heathcheck_service, grasp_planning_service
from api.exceptions import ApiException

app_blueprint = Blueprint("app_blueprint", __name__)


@app_blueprint.route("/", methods=["GET"])
def heathcheck_service_api():
    return jsonify(heathcheck_service())


@app_blueprint.route("/grasp-planning", methods=["POST"])
def grasp_planning_api():
    return jsonify(grasp_planning_service(**dict(request.form)))


@app_blueprint.errorhandler(ApiException)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response
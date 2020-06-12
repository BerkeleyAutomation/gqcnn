from flask import Blueprint, jsonify

from api.service import heathcheck_service, grasp_planning_service

app_blueprint = Blueprint("app_blueprint", __name__)


@app_blueprint.route("/", methods=["GET"])
def heathcheck_service_api():
    return jsonify(heathcheck_service())


@app_blueprint.route("/grasp-planning", methods=["POST"])
def grasp_planning_api():
    return jsonify(grasp_planning_service())
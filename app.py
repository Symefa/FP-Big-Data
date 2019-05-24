from flask import Flask, request
import logging
from engine import ClusteringEngine
import json
from flask import Blueprint, render_template

main = Blueprint('main', __name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/model1/<int:cbg_id>/cluster", methods=["GET"])
def get_cluster1(cbg_id):
    logger.debug("ID %s requested", cbg_id)
    cluster_category = clustering_engine.get_cluster1(cbg_id)
    return json.dumps(cluster_category)


@main.route("/model2/<int:cbg_id>/cluster", methods=["GET"])
def get_cluster2(cbg_id):
    logger.debug("ID %s requested", cbg_id)
    cluster_category = clustering_engine.get_cluster2(cbg_id)
    return json.dumps(cluster_category)


@main.route("/model3/<int:cbg_id>/cluster", methods=["GET"])
def get_cluster3(cbg_id):
    logger.debug("ID %s requested", cbg_id)
    cluster_category = clustering_engine.get_cluster3(cbg_id)
    return json.dumps(cluster_category)


def create_app(spark_session, dataset_path):
    global clustering_engine

    clustering_engine = ClusteringEngine(spark_session, dataset_path)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app

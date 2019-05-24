import os
import logging
import pandas as pd
from pyspark.sql import Row
from pyspark.sql import types
from pyspark.sql.functions import explode
import pyspark.sql.functions as func
from sklearn.preprocessing import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringEngine:
    """ cbg Clusters
    """

    def __transform_model(self):
        """Train the ALS model with the current dataset
        """
        logger.info("Transforming model 1...")
        self.df_cbg1 = self.df_cbg1.withColumn(
            "latitude", self.df_cbg1["latitude"].cast("double"))
        self.df_cbg1 = self.df_cbg1.withColumn(
            "longitude", self.df_cbg1["longitude"].cast("double"))
        assembler = VectorAssembler(
            inputCols=["latitude", "longitude"], outputCol='features')
        self.df_cbg1 = assembler.setHandleInvalid(
            "skip").transform(self.df_cbg1)
        logger.info("Done transforming!")

        logger.info("Transforming model 2...")
        self.df_cbg2 = self.df_cbg2.withColumn(
            "latitude", self.df_cbg2["latitude"].cast("double"))
        self.df_cbg2 = self.df_cbg2.withColumn(
            "longitude", self.df_cbg2["longitude"].cast("double"))
        assembler = VectorAssembler(
            inputCols=["latitude", "longitude"], outputCol='features')
        self.df_cbg2 = assembler.setHandleInvalid(
            "skip").transform(self.df_cbg2)
        logger.info("Done transforming!")

        logger.info("Transforming model 3...")
        self.df_cbg3 = self.df_cbg3.withColumn(
            "latitude", self.df_cbg3["latitude"].cast("double"))
        self.df_cbg3 = self.df_cbg3.withColumn(
            "longitude", self.df_cbg3["longitude"].cast("double"))
        assembler = VectorAssembler(
            inputCols=["latitude", "longitude"], outputCol='features')
        self.df_cbg3 = assembler.setHandleInvalid(
            "skip").transform(self.df_cbg3)
        logger.info("Done transforming!")

    def __train_model(self):
        """Train the ALS model with the current dataset
        """
        logger.info("Training model 1...")
        kmeans_1 = KMeans().setK(9).setSeed(1)
        model_1 = kmeans_1.fit(self.df_cbg1)
        logger.info("Model 1 built!")
        logger.info("Evaluating the model 1...")
        self.predictions_1 = model_1.transform(self.df_cbg1)
        logger.info("Model 1 Done !")

        logger.info("Training model 2...")
        kmeans_2 = KMeans().setK(9).setSeed(1)
        model_2 = kmeans_2.fit(self.df_cbg2)
        logger.info("Model 2 built!")
        logger.info("Evaluating the model 2...")
        self.predictions_2 = model_2.transform(self.df_cbg2)
        logger.info("Model 2 Done !")

        logger.info("Training model 3...")
        kmeans_3 = KMeans().setK(9).setSeed(1)
        model_3 = kmeans_3.fit(self.df_cbg3)
        logger.info("Model 3 built!")
        logger.info("Evaluating the model 3...")
        self.predictions_3 = model_3.transform(self.df_cbg3)
        logger.info("Model 3 Done !")

    def get_cluster1(self, census_block_group):
        """
        """
        pred = self.predictions_1.filter(
            self.predictions_1['census_block_group'] == census_block_group)
        pred = pred.toPandas()
        pred = pred.to_json()
        return pred

    def get_cluster2(self, census_block_group):
        """
        """
        pred = self.predictions_2.filter(
            self.predictions_2['census_block_group'] == census_block_group)
        pred = pred.toPandas()
        pred = pred.to_json()
        return pred

    def get_cluster3(self, census_block_group):
        """
        """
        pred = self.predictions_3.filter(
            self.predictions_3['census_block_group'] == census_block_group)
        pred = pred.toPandas()
        pred = pred.to_json()
        return pred

    def __init__(self, spark_session, dataset_path):
        """Init the recommendation engine given a Spark context and a dataset path
        """
        logger.info("Starting up the Clustering Engine: ")
        self.spark_session = spark_session
        # Load ratings data for later use
        logger.info("Loading CBG data...")
        file_name1 = 'model-1.txt'
        dataset_file_path1 = os.path.join(dataset_path, file_name1)
        exist = os.path.isfile(dataset_file_path1)
        if exist:
            self.df_cbg1 = spark_session.read.csv(
                dataset_file_path1, header=None, inferSchema=True)
            self.df_cbg1 = self.df_cbg1.selectExpr(
                "_c0 as census_block_group", "_c1 as raw_visitor_count", "_c2 as latitude", "_c3 as longitude")

        file_name2 = 'model-2.txt'
        dataset_file_path2 = os.path.join(dataset_path, file_name2)
        exist = os.path.isfile(dataset_file_path2)
        if exist:
            self.df_cbg2 = spark_session.read.csv(
                dataset_file_path2, header=None, inferSchema=True)
            self.df_cbg2 = self.df_cbg2.selectExpr(
                "_c0 as census_block_group", "_c1 as raw_visitor_count", "_c2 as latitude", "_c3 as longitude")

        file_name3 = 'model-3.txt'
        dataset_file_path3 = os.path.join(dataset_path, file_name3)
        exist = os.path.isfile(dataset_file_path3)
        if exist:
            self.df_cbg3 = spark_session.read.csv(
                dataset_file_path3, header=None, inferSchema=True)
            self.df_cbg3 = self.df_cbg3.selectExpr(
                "_c0 as census_block_group", "_c1 as raw_visitor_count", "_c2 as latitude", "_c3 as longitude")

        # Train the model
        self.__transform_model()
        self.__train_model()

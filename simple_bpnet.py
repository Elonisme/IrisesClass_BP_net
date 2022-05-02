# -*- coding =utf-8 -*-
# @Time : 2022-04-26 20:00
# @Author : Elon
# @File : simple_bpnet.py
# @Software : PyCharm
# -*- coding =utf-8 -*-
# @Time : 2022-04-25 8:55
# @Author : Elon
# @File : simple_bpnet.py
# @Software : PyCharm
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand, to_csv
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
import torch
import torch.nn as nn
from bpnet import Net


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Irises") \
        .master('local[6]').config('spark.driver.memory', '1g') \
        .getOrCreate()

    # Read in mnist_train.csv dataset
    df = spark.read.option("inferSchema", "true").csv('Irises.csv').orderBy(rand()).repartition(6)
    print(df.show(151))
    print(df.columns)
    print(df.count())
    network = Net()
    print(network)

    # Build the pytorch object
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.02
    )

    # Setup features
    vector_assembler = VectorAssembler(inputCols=df.columns[1:5], outputCol='features')
    print(df.columns[1:5])

    # Demonstration of some options. Not all are required
    # Note: This uses the barrier execution mode, which is sensitive to the number of partitions
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='_c0',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=70,
        verbose=1,
        validationPct=0.2
    )

    # Create and save the Pipeline
    p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
    p.write().overwrite().save('simple_bp')

    # Example of loading the pipeline
    loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_bp'))

    # Run predictions and evaluation
    predictions = loaded_pipeline.transform(df).persist()
    print(type(predictions))
    print(predictions.show(151))
    pands_pre = predictions.toPandas()
    pands_pre.to_csv("predictions.csv", index_label="index_label")

    # predictions = predictions.rdd.map(to_csv).toDF(['sepal','length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','predictions'])
    # predictions.write.csv('./predictions_data',header=True)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="_c0", predictionCol="predictions", metricName="accuracy")
    print(type(evaluator))


    accuracy = evaluator.evaluate(predictions)
    print("Train accuracy = %g" % accuracy)


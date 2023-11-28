from dataclasses import dataclass
from typing import List, Sequence, Optional

from loguru import logger

import utils.time_utils
from pipeline.analysis.aggregate.aggregate_analyzer import AggregateAnalysis, AggregateAnalyzer
from pipeline.analysis.datasets.dataset_analyzer import DatasetAnalysis, DatasetAnalyzer
from pipeline.analysis.models.model_evaluator import ModelEvaluation, ModelEvaluator
from pipeline.dataset_providers.base import DatasetProvider
from pipeline.feature_engineering.base import FeatureEngineering, Dataset
from pipeline.model_builder import ModelData, ModelBuilder


@dataclass(frozen=True)
class PipelineResults:
    models: List[ModelData]
    dataset_analysis: List[DatasetAnalysis]
    model_evaluations: List[ModelEvaluation]
    aggregate_analysis: Optional[AggregateAnalysis]


class Pipeline:
    """
    Class that handles executing a pipeline that consists of:
    1. feature engineering
    2. data analysis
    3. model training
    4. model evaluation

    The results of each stage are returned after the pipeline has executed
    """

    def __init__(
            self,
            dataset_provider: DatasetProvider,
            feature_engineering_pipeline: Sequence[FeatureEngineering],
            dataset_analyzers: Sequence[DatasetAnalyzer],
            model_builder: ModelBuilder,
            model_analyzers: Sequence[ModelEvaluator],
            aggregate_analyzer: Optional[AggregateAnalyzer] = None
    ):
        """
        :param dataset_provider:
        :param feature_engineering_pipeline: components that apply some form of feature engineering to the dataset,
            executed in the same order as they are in the provided sequence
        :param dataset_analyzers:
        :param model_builder:
        :param model_analyzers:
        """

        self.dataset_provider = dataset_provider
        self.feature_engineering_pipeline = feature_engineering_pipeline
        self.dataset_analyzers = dataset_analyzers
        self.model_builder = model_builder
        self.model_evaluators = model_analyzers
        self.aggregate_analyzer = aggregate_analyzer

    def execute(self) -> PipelineResults:
        logger.info("Starting pipeline execution...")

        data_analysis_results: List[DatasetAnalysis] = []
        models: List[ModelData] = []
        model_evals: List[ModelEvaluation] = []

        train = self.dataset_provider.get_training_dataset()
        test = self.dataset_provider.get_testing_dataset()

        logger.info("Dataset retrieved. Executing feature engineering pipeline...")
        engineered_train, engineered_test = train, test
        for fe in self.feature_engineering_pipeline:
            engineered_train, engineered_test = fe.engineer(engineered_train, engineered_test)

        logger.info("Feature engineering pipeline executed. Analyzing dataset...")
        data_analysis_results.extend([
            _analyze_dataset(da, train)
            for da in self.dataset_analyzers
        ])

        for model in self.model_builder.build(engineered_train, engineered_test):
            logger.info("Model trained. Analyzing model...")
            model.training_dataset = train
            model.testing_dataset = test
            model.training_dataset_engineered = engineered_train
            model.testing_dataset_engineered = engineered_test

            models.append(model)

            temp_res = [
                _analyze_model(ma, model)
                for ma in self.model_evaluators
            ]
            model_evals.extend([
                x for x in temp_res if x is not None
            ])

        logger.info("Models analyzed successfully. Performing aggregate analysis...")
        aggregate_analysis = self.aggregate_analyzer.analyze(model_evals) if self.aggregate_analyzer is not None else None

        logger.info("Pipeline terminated successfully, returning results...")
        return PipelineResults(
            dataset_analysis=data_analysis_results,
            models=models,
            model_evaluations=model_evals,
            aggregate_analysis=aggregate_analysis,
        )


@utils.time_utils.print_time_perf
def _analyze_dataset(da: DatasetAnalyzer, dataset: Dataset):
    return da.analyze(dataset)


@utils.time_utils.print_time_perf
def _analyze_model(ma: ModelEvaluator, model: ModelData):
    return ma.evaluate(model)

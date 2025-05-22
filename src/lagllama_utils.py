import os
import torch
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
torch.serialization.add_safe_globals([NegativeLogLikelihood])
torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])

os.chdir("lag-llama/lag-llama") # navigating into the lag-llama repo
from lag_llama.gluon.estimator import LagLlamaEstimator


def prep_gluonts_df(df, freq = "H"):
    start_index = df["date"].min()
    data = [{FieldName.START:  pd.Timestamp(start_index),
             FieldName.TARGET:  df["y"].values,}]
    return ListDataset(data, freq = freq)


def lagllama_estimator(dataset,
                        prediction_length,
                        device,
                        context_length = 32,
                        rope_scaling = True,
                        num_samples = 100):
    
    ckpt = torch.load("lag-llama/lag-llama.ckpt", map_location = device, weights_only = False)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    rope_scaling_arg = {"type": "linear", "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"])} if rope_scaling else None

    estimator = LagLlamaEstimator(ckpt_path = "lag-llama/lag-llama.ckpt",
                                  context_length = context_length,
                                  prediction_length = prediction_length,
                                  device = device,
                                  input_size = estimator_args["input_size"],
                                  n_layer = estimator_args["n_layer"],
                                  n_embd_per_head = estimator_args["n_embd_per_head"],
                                  n_head = estimator_args["n_head"],
                                  scaling = estimator_args["scaling"],
                                  time_feat = estimator_args["time_feat"],
                                  rope_scaling = rope_scaling_arg,)

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_iterator, actual_series_iterator = make_evaluation_predictions(dataset = dataset,
                                                                            predictor = predictor,
                                                                            num_samples = num_samples)

    return list(forecast_iterator), list(actual_series_iterator)


def extract_hyperparams_lagllama(hyperparams, horizon):
    mask = hyperparams["horizon"] == horizon
    rope_scaling = hyperparams.loc[mask, "rope_scaling"]
    context_length = hyperparams.loc[mask, "context_length"]
    rope_scaling, context_length = rope_scaling.values[0], context_length.values[0]
    return rope_scaling, context_length
    
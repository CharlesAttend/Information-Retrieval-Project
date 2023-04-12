import ir_measures
from ir_measures import nDCG, MAP, MRR
import pandas as pd
from src.dataloader import IrDataset


def eval(dataset: IrDataset, run: pd.DataFrame):
    """Eval the run on three metrics : nDCG@10, MAP, MRR@10

    Parameters
    ----------
    dataset : IrDataset
        The IrDataset instance used by the model
    run : pandas.DataFrame
        _description_
    """
    qrels = dataset.qrels_iter
    results = []
    for metric in ir_measures.iter_calc([nDCG @ 10, MAP@1000, MRR @ 10], qrels, run):
        results.append((results, metric))
    return pd.DataFrame(results)


def save_and_agregate_result(results: pd.DataFrame, model_name):
    """Compute and save result

    Parameters
    ----------
    results : pd.DataFrame
        _description_
    """
    results.groupby(by="measure").mean().to_csv(f"output/{model_name}.csv")

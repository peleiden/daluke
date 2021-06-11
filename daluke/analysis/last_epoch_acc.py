
import os

import click

from pelutils.logger import log

from daluke.analysis.pretrain import TrainResults

@click.command()
@click.argument("location")
@click.option("--out", default=None, type=str)
def log_last_epoch_acc(location: str, out: str):
    log.configure(os.path.join(out if out is not None else location, "last-epoch-acc.log"), "Last epoch scores of pretraining model")
    res = TrainResults.load(location)
    e = res.epoch
    for i, k in enumerate(res.top_k):
        log(f"K={k}")
        w_acc, e_acc = res.w_accuracies[e, :, i], res.e_accuracies[e, :, i]
        log(f"Word:   {100*w_acc.mean():.3f}", with_info=False)
        log(f"Entity: {100*e_acc.mean():.3f}", with_info=False)

if __name__ == "__main__":
    with log.log_errors:
        log_last_epoch_acc()

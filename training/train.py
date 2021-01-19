from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl

import littissuesegmentation as lts

pl.seed_everything(2020)
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# add PROGRAM level args
parser.add_argument(
    "--run_name",
    type=str,
    default=f"TissueSegmentation",
    help="Name to be given to the run (logging)",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="TissueSegmentation",
    help="Wandb project to log to, check out wandb... please",
)
parser.add_argument(
    "--terminator_patience",
    type=int,
    default="5",
    help="Patience for early termination",
)

# add model specific args
parser = lts.models.LitSMPModel.add_model_specific_args(parser)

# add data specific args
parser = lts.dataloaders.LitSMPDatamodule.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(precision=16, gpus=1)

# This is necessary to add the defaults when calling the help option
parser = ArgumentParser(
    parents=[parser],
    add_help=False,
    formatter_class=ArgumentDefaultsHelpFormatter,
)

if __name__ == "__main__":

    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    model = lts.models.ARCHITECTURES[dict_args["architecture"]](**dict_args)
    lts.train.main_train(model, args)

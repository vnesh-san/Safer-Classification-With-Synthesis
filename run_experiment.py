import importlib

import pytorch_lightning as pl
from util import ImageGeneratingLogger

from dataloader import *
from network import RobustClassifier

def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    # Hide lines below until Lab 5
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default='safer_classification_with_synthesis')
    parser.add_argument("--run_name", type=str, default='Siamese Frozen')

    # Hide lines above until Lab 5

    lit_model_group = parser.add_argument_group("LitModel Args")
    RobustClassifier.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    model = RobustClassifier(args)
    data = MNIST(args)
    logger = pl.loggers.WandbLogger(project=args.project_name, name=args.run_name)
    logger.log_hyperparams(vars(args))
    logger.watch(model)
    
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    sample_generator = ImageGeneratingLogger(5)
    callbacks = [early_stopping_callback, model_checkpoint_callback, sample_generator]
    # callbacks = [sample_generator]
    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, gpus=[args.gpus], callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    trainer.tune(model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

if __name__ == '__main__':
    main()
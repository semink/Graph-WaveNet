from STNET.dataset import DataModule

from argparse import ArgumentParser
import yaml

from supervisor import GWNETPredictor

from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer


def data_load(dataset, batch_size, seq_len, horizon, **kwargs):
    dm = DataModule(dataset, batch_size, seq_len,
                    horizon)
    dm.prepare_data()
    return dm


def train_model(args):
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    dm = data_load(**config['HPARAMS']['DATA'])
    model = GWNETPredictor(hparams=config['HPARAMS'],
                           scaler=dm.get_scaler(), A=dm.get_adj())

    logger = TensorBoardLogger(**config['LOG'], default_hp_metric=False)
    trainer = Trainer(
        **config['TRAINER'], callbacks=[RichModelSummary(config['MODEL_SUMMARY']['max_depth']),
                                        RichProgressBar(),
                                        ModelCheckpoint(monitor=config['HPARAMS']['OPTIMIZER']['monitor_metric'])],
        logger=logger)

    trainer.fit(model, dm)
    result = trainer.test(model, dm, ckpt_path='best')


def test_model(args):
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    dm = data_load(**config['HPARAMS']['DATA'])
    model = GWNETPredictor.load_from_checkpoint(
        config['TEST']['checkpoint'], scaler=dm.get_scaler(), A=dm.get_adj(), map_location='cpu')

    trainer = Trainer(**config['TRAINER'],
                      callbacks=[RichProgressBar()],
                      enable_checkpointing=False,
                      logger=False)
    result = trainer.test(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Program specific args
    parser.add_argument("--config", type=str,
                        default="config/la.yaml", help="Configuration file path")
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')

    args = parser.parse_args()

    assert (
        not args.train) | (
        not args.test), "Only one of --train and --test flags can be turned on."
    assert (
        args.train) | (
        args.test), "At least one of --train and --test flags must be turned on."

    if args.train:
        train_model(args=args)
    elif args.test:
        test_model(args=args)

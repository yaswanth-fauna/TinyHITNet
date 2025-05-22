import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models import build_model
from dataset import build_dataset
from metrics.epe import EPEMetric
from metrics.rate import RateMetric
from torchmetrics import MetricCollection


class EvalModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(self.hparams)
        self.max_disp = self.hparams.max_disp
        self.metric = MetricCollection(
            {
                "epe": EPEMetric(),
                "rate_1e-1": RateMetric(0.1),
                "rate_1": RateMetric(1.0),
                "rate_3": RateMetric(3.0),
            }
        )

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)

    def test_step(self, batch, batch_idx):
        pred = self(batch["left"], batch["right"])
        mask = (batch["disp"] < self.max_disp) & (batch["disp"] > 1e-3)
        self.metric(pred, batch["disp"], mask)
        return

    def on_test_epoch_end(self):
        print(self.metric.compute())
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--data_type_val", type=str, nargs="+")
    parser.add_argument("--data_root_val", type=str, nargs="+")
    parser.add_argument("--data_list_val", type=str, nargs="+")
    parser.add_argument("--data_size_val", type=int, nargs=2, default=None)
    parser.add_argument("--data_augmentation", type=int, default=0)
    parser.add_argument("--eval_metrics", action="store_true")
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--onnx_file_name", type=str, default="super_resolution.onnx")
    args = parser.parse_args()

    model = EvalModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt, weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    
    if args.export_onnx:
        print("Exporting to ONNX format...")
        # Input to the model
        x = torch.randn(1, 3, 360, 640, requires_grad=False)
        torch_out = model(x, x)

        # Export the model
        torch.onnx.export(model,               # model being run
                        (x, x),                         # model input (or a tuple for multiple inputs)
                        args.onnx_file_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=17,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['left', 'right'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'left' : {0 : 'batch_size'},
                                      'right' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
        

    if args.eval_metrics:
        print("Loading dataset...")
        dataset = build_dataset(args, training=False)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
        )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=False,
        )
        trainer.test(model, loader)

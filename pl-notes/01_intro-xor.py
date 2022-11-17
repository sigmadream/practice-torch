import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

xor_input = [
    Variable(torch.Tensor([0, 0])),
    Variable(torch.Tensor([0, 1])),
    Variable(torch.Tensor([1, 0])),
    Variable(torch.Tensor([1, 1])),
]

xor_targets = [
    Variable(torch.Tensor([0])),
    Variable(torch.Tensor([1])),
    Variable(torch.Tensor([1])),
    Variable(torch.Tensor([0])),
]


class XORModel(pl.LightningModule):
    def __init__(self) -> None:
        super(XORModel, self).__init__()
        self.input_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, input):
        x = self.input_layer(input)
        x = self.sigmoid(x)
        output = self.output_layer(x)
        return output

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        outputs = self(xor_input)
        loss = self.loss(outputs, xor_target)
        return loss


if __name__ == "__main__":
    print(f"{torch.__version__}\t{pl.__version__}")
    xor_data = list(zip(xor_input, xor_targets))
    train_loader = DataLoader(xor_data, batch_size=1000)
    checkpoint_callback = ModelCheckpoint()
    model = XORModel()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, train_dataloaders=train_loader)
    print(checkpoint_callback.best_model_path)
    train_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    test = torch.utils.data.DataLoader(xor_input, batch_size=1)
    for val in xor_input:
        _ = train_model(val)
        print([int(val[0]), int(val[1])], int(_.round()))

    train_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    total_accuracy = []
    for xor_input, xor_target in train_loader:
        for i in range(100):
            output_tensor = train_model(xor_input)
            test_accuracy = accuracy(output_tensor, xor_target.int())
            total_accuracy.append(test_accuracy)
    total_accuracy = torch.mean(torch.stack(total_accuracy))
    print("TOTAL ACCURACY FOR 100 ITERATIONS: ", total_accuracy.item())

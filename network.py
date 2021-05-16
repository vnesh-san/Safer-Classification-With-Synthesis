import torch
import wandb
import numpy as np
import torch.optim as optim
import argparse
import torch.nn as nn
from util import Accuracy, ContrastiveLoss
from pytorch_lightning import LightningModule
from model import SiameseNet, Generator

class RobustClassifier(LightningModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.num_classes = args.num_classes
        self.latent_dim = args.latent_dim
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.lr = args.lr

        self.class_noise_convertor = nn.ModuleDict({str(k): nn.Sequential(nn.Linear(self.feature_dim, self.latent_dim),
                                                                          nn.ReLU(),
                                                                          nn.Linear(self.latent_dim, self.latent_dim)).to(self.device) for k in range(self.num_classes)})

        self.generator = Generator(ngpu=1)
        self.generator.load_state_dict(torch.load(args.gen_weights))
        self.class_identifier = SiameseNet()
        self.class_identifier.load_state_dict(torch.load(args.siamese_weights))
        if args.generator_pre_train:
            self.generator.to(self.device).freeze()
        if args.siamese_pre_train:
            self.class_identifier.to(self.device).freeze()

        # self.class_similarity = nn.Linear(4096, 1)
        self.creterion = ContrastiveLoss(1)
        self.threshold = args.threshold

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--num_classes", type=int, default=10, help="Number of Classes")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--latent_dim", type=int, default=100)
        parser.add_argument("--feature_dim", type=int, default=100)
        parser.add_argument("--gen_weights", type=str, default="weights/gen_weights.pth")
        parser.add_argument("--siamese_weights", type=str, default="weights/siamese_weights.pth")
        parser.add_argument("--generator_pre_train", dest='generator_pre_train', default=True, action='store_true')
        parser.add_argument("--no_generator_pre_train", dest='generator_pre_train', default=True, action='store_false')
        parser.set_defaults(generator_pre_train=True)
        parser.add_argument("--siamese_pre_train", dest='siamese_pre_train', default=True, action='store_true')
        parser.add_argument("--no_siamese_pre_train", dest='siamese_pre_train', default=True, action='store_false')
        parser.set_defaults(siamese_pre_train=True)
        return parser


    def forward(self, x):
        batch_size = x.size(0)
        embeddings1, embeddings2 = torch.Tensor([]), torch.Tensor([])
        scores = torch.ones(self.num_classes, batch_size)
        noise = torch.rand(batch_size, self.feature_dim, device=self.device)
        for class_idx, model in self.class_noise_convertor.items():
            class_noise = model(noise).view(batch_size, -1, 1, 1)
            gen_imgs = self.generator(class_noise)
            embed1, embed2 = self.class_identifier(gen_imgs, x)
            embeddings1 = torch.cat((embeddings1.to(self.device), embed1.to(self.device)), dim=0)
            embeddings2 = torch.cat((embeddings2.to(self.device), embed2.to(self.device)), dim=0)
            scores[int(class_idx)] = nn.functional.cosine_similarity(embed1, embed2, dim=1)
        self.register_buffer("embeddings_1", embeddings1.view(-1, 4096))
        self.register_buffer("embeddings_2", embeddings2.view(-1, 4096))
        scores = torch.softmax(scores[scores > 0].view(batch_size, -1), dim=1)
        pred = torch.argmax(scores, dim=1).to(self.device)#torch.Tensor([torch.argmax(img_score) if (img_score.max()-img_score.min())>0.5 else -1 for img_score in scores]).to(self.device)
        return pred

    def calculate_loss(self, y_true):
        siamese_labels = torch.Tensor([])
        for class_idx in range(self.num_classes):
            temp = torch.from_numpy(np.where(y_true.cpu().numpy() == class_idx, 1, 0))
            siamese_labels = torch.cat((siamese_labels.to(self.device), temp.to(self.device)), dim=0)
        result = self.creterion(self.embeddings_1, self.embeddings_2, siamese_labels)
        return result

    def training_step(self, batch, batch_idx):
        img, y_true = batch
        img, y_true = img.to(self.device), y_true.to(self.device)
        y_pred = self(img)
        result = self.calculate_loss(y_true)
        self.train_acc(y_pred, y_true)
        self.log('train_acc', self.train_acc, on_epoch=True, on_step=False)
        self.log('train_loss', result, on_step=True)

        return result

    def validation_step(self, batch, batch_idx):
        img, y_true = batch
        img, y_true = img.to(self.device), y_true.to(self.device)
        y_pred = self(img)
        val_loss = self.calculate_loss(y_true)
        self.log('val_loss', val_loss, prog_bar=True)
        self.val_acc(y_pred, y_true)
        self.log('val_acc', self.val_acc, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        img, y_true = batch
        img, y_true = img.to(self.device), y_true.to(self.device)
        y_pred = self(img)
        self.test_acc(y_pred, y_true)
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad_, self.parameters()), lr=self.lr)
        return [optimizer], []
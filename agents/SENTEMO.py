import numpy as np
import logging
from tqdm import tqdm
import shutil
import random
import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
from agents.base import BaseAgent
from graphs.models.TRE_Encoder import Text_Encoder
from data_loader.Text_loader import TextDataLoader
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, accuracy, Evaluate
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class SENTEMO_Agent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define data_loader
        self.data_loader = TextDataLoader(config)
        
        # define models
        self.model = Text_Encoder(config)

        # define loss
        self.loss = nn.NLLLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # Define Scheduler
        #lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.max_epoch)), 0.9)
        #self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min')
        # initialize my counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_accuracy = 0

        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed_all(self.config.seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()

        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")
        self.config.device = self.device
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)


    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            #'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
    
    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            #self.scheduler.step(epoch)
            self.train_one_epoch()

            valid_accuracy , valid_loss = self.validate()
            self.scheduler.step(valid_loss)

            is_best = valid_accuracy > self.best_valid_accuracy
            if is_best:
                self.best_valid_accuracy = valid_accuracy

            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        
        self.model.train()

        # Initialize your average meters
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        for itr, (x, y, _) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # model
            pred = self.model(x.permute(1 ,0))
            # loss
            y = torch.max(y, 1)[1]
            y = y.long()
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            
            epoch_loss.update(cur_loss.item())
            batch_accuracy = accuracy(y, pred)
            epoch_acc.update(batch_accuracy, x.size(0))

            self.current_iteration += 1

        self.summary_writer.add_scalar("epoch-training/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/accuracy", epoch_acc.val, self.current_iteration)
        tqdm_batch.close()

        print( "Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(epoch_loss.val) + " - acc-: " + str(epoch_acc.val ))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                            desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in evaluation mode
        self.model.eval()

        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()
        
        for itr, (x, y, _) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.pin_memory().cuda(non_blocking=True), y.cuda(non_blocking=True)
            # model
            pred = self.model(x.permute(1 ,0))
            # loss
            y = torch.max(y, 1)[1]
            y = y.long()
            cur_loss = self.loss(pred, y)

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during Validation.')
            
            batch_accuracy = accuracy(y, pred)
            epoch_acc.update(batch_accuracy)
            epoch_loss.update(cur_loss.item())

        self.summary_writer.add_scalar("epoch_validation/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/accuracy", epoch_acc.val , self.current_iteration)

        print("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(epoch_loss.val) + " - acc-: " + str(epoch_acc.val))
        tqdm_batch.close()

        return epoch_acc.val, epoch_loss.val

    def test(self):
        """
        Main test loop
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                    desc="Testing at -{}-".format(self.current_epoch))

        # set the model in evaluation mode
        self.model.eval()
        evaluator = Evaluate()
        epoch_acc = AverageMeter()
        
        with torch.no_grad():
            final_predictions = []
            all_predictions = []
            x_raw = []
            y_raw = []
            for itr, (x, y, _) in enumerate(tqdm_batch):

                x_raw = x_raw + [x for x in x]
                y_raw = y_raw + [y for y in y]
                
                if self.cuda:
                    x, y = x.pin_memory().cuda(non_blocking=True), y.cuda(non_blocking=True)
                # model
                pred = self.model(x.permute(1 ,0))
                all_predictions.append(pred)
                # loss
                y = torch.max(y, 1)[1]
                y = y.long()
                
                batch_accuracy = accuracy(y, pred)
                epoch_acc.update(batch_accuracy)

            print("Testing results  -- acc-: " + str(epoch_acc.val))
            tqdm_batch.close()
            for p in all_predictions:
                for sub_p in p:
                    final_predictions.append(sub_p.cpu().detach().numpy())
                    
            predictions = [np.argmax(p).item() for p in final_predictions]
            targets = [np.argmax(t).item() for t in y_raw]
            correct_predictions = float(np.sum(predictions == targets))

            # predictions
            predictions_human_readable = ((x_raw, predictions))
            # actual targets
            target_human_readable = ((x_raw,  targets))

            if self.config.data_type == 'SENTEMO':
                emotion_dict = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
            elif self.config.data_type == "SEM_EVAL_OC" or "SEM_EVAL_OC_Translated":
                emotion_dict = {0: 'anger', 1: 'joy', 2: 'fear', 3: 'sadness'}
            # convert results into dataframe
            model_test_result = pd.DataFrame(predictions_human_readable[1],columns=["emotion"])
            test = pd.DataFrame(target_human_readable[1], columns=["emotion"])

            model_test_result.emotion = model_test_result.emotion.map(lambda x: emotion_dict[int(float(x))])
            test.emotion = test.emotion.map(lambda x: emotion_dict[int(x)])

            evaluator.evaluate_class(model_test_result.emotion, test.emotion )

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()


from tqdm import tqdm


class TimeClassifierTrainer:

    def __init__(self, train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, early_stoper, EPOCH):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stoper = early_stoper
        self.EPOCH = EPOCH

    def train(self):
        train_losses = []
        val_losses = []
        for i in range(self.EPOCH):

            train_loss = 0
            val_loss = 0

            self.model.train()
            for x, label in tqdm(self.train_dataloader, desc="Training"):
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()*len(x) / len(self.train_dataloader)
            self.scheduler.step()
            train_losses.append(train_loss)

            self.model.eval()
            for x, label in tqdm(self.val_dataloader, desc="Validating"):
                pred = self.model(x)
                loss = self.criterion(pred, label)
                val_loss += loss.item()*len(x) / len(self.val_dataloader)
            val_losses.append(val_loss)

            print("Epoch {0}/{1}. Train loss {2:.4}. Val loss {3:.4}".format(i + 1, self.EPOCH, train_loss, val_loss))

            self.early_stoper(val_loss, self.model)
            if self.early_stoper.early_stop:
                print("Early stop is processed.")
                break

        self.train_losses = train_losses
        self.val_losses = val_losses




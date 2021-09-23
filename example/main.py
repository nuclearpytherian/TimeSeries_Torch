# Main

from utils.utils import PandasTimeClassifierDataset, random_pandasDF
from modules.models import TimeClassifierLSTMNet
from modules.train import TimeClassifierTrainer
from modules.scheduler import CosineAnnealingWarmUpRestarts
from modules.earlystop import EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader

if __name__ == "__main__":

    # Data set up
    train_data_df = random_pandasDF(N=480, N_FEATURES=30, NUM_CLASSES=3)
    val_data_df = random_pandasDF(N=120, N_FEATURES=30, NUM_CLASSES=3)
    train_dataset = PandasTimeClassifierDataset(train_data_df, label_col='y', TIME_STEP=20)
    val_dataset = PandasTimeClassifierDataset(val_data_df, label_col='y', TIME_STEP=20)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # model set up
    model = TimeClassifierLSTMNet(INPUT_DIM=30, HIDDEN_DIM=64, N_LAYER=1, OUTPUT_DIM=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1, T_up=10, gamma=0.5)
    early_stopping = EarlyStopping(patience=2, verbose=False, path="Trained_TS_model.pt")

    # Train set up
    Training = TimeClassifierTrainer(train_dataloader=train_dataloader,
                                     val_dataloader=val_dataloader,
                                     model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     early_stoper=early_stopping,
                                     EPOCH=5)
    # Run training
    Training.train()




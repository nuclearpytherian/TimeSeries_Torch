# Main

from utils.utils import PandasTimeClassifierDataset, random_pandasDF
from modules.models import TimeClassifierConv1dLSTMNet
from modules.train import TimeClassifierTrainer
from modules.scheduler import CosineAnnealingWarmUpRestarts
from modules.earlystop import EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Parameters
    TIME_STEP = 20
    N_FEATURES = 10
    NUM_CLASSES = 2
    EPOCH = 10
    # Data set up
    train_data_df, val_data_df = random_pandasDF(batch_size=100, N_FEATURES=N_FEATURES, NUM_CLASSES=NUM_CLASSES)
    train_dataset = PandasTimeClassifierDataset(train_data_df, label_col='y', TIME_STEP=TIME_STEP)
    val_dataset = PandasTimeClassifierDataset(val_data_df, label_col='y', TIME_STEP=TIME_STEP)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model set up
    model = TimeClassifierConv1dLSTMNet(INPUT_DIM=N_FEATURES,TIME_STEP=TIME_STEP, HIDDEN_DIM=64, N_LAYER=1, OUTPUT_DIM=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1, T_up=10, gamma=0.5)
    early_stopping = EarlyStopping(patience=10, verbose=False, path="Earlystop_TS_model.pt")

    # Train set up
    Training = TimeClassifierTrainer(train_dataloader=train_dataloader,
                                     val_dataloader=val_dataloader,
                                     model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scheduler=None,
                                     early_stoper=early_stopping,
                                     EPOCH=EPOCH)
    # Run training
    Training.train()

    # Save model
    Training.save_model(model_path='saved_model.pt')




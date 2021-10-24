# Main

from utils.utils import PandasTimeDataset, random_pandasDFClassifier, Predictor
from modules.models import TimeConv1dLSTMNetClassifier
from modules.train import Trainer
from modules.evaluation import Evaluator
from torch.optim.lr_scheduler import StepLR
from modules.earlystop import EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    # Parameters
    TIME_STEP = 20
    N_FEATURES = 10
    NUM_CLASSES = 4
    EPOCH = 2

    # Data set up
    train_data_df, val_data_df = random_pandasDFClassifier(batch_size=100, N_FEATURES=N_FEATURES, NUM_CLASSES=NUM_CLASSES)
    train_dataset = PandasTimeDataset(train_data_df, label_col='y', TIME_STEP=TIME_STEP, mode='classifier')
    val_dataset = PandasTimeDataset(val_data_df, label_col='y', TIME_STEP=TIME_STEP, mode='classifier')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeConv1dLSTMNetClassifier(INPUT_DIM=N_FEATURES,
                                        TIME_STEP=TIME_STEP,
                                        HIDDEN_DIM=64,
                                        N_LAYER=1,
                                        OUTPUT_DIM=NUM_CLASSES,
                                        DROPOUT=0.0,
                                        bidirectional=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    early_stopping = EarlyStopping(patience=10, verbose=False)

    # Train set up
    Training = Trainer(train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        early_stoper=early_stopping,
                        EPOCH=EPOCH,
                       device=device)

    # Run training
    Training.train()

    # Plot loss graph
    Training.plot_loss_graph()

    # Save model
    best_model = True
    Training.save_model(best_model=best_model)

    # Evaluation
    if best_model:
        model_path = 'artifact/best_epoch_model.pt'
    else:
        model_path = 'artifact/last_epoch_model.pt'
    evaluator = Evaluator(model=model,
                          PATH=model_path,
                          device_type='cpu')
    evaluator(train_dataloader)
    label_dic = {0:'Type I', 1:'Type II', 2:'Type III', 3:'Type IV'}
    evaluator.model_plot(train_dataloader, dim=3, markersize=2, label_dict=label_dic)



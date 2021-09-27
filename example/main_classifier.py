# Main

from utils.utils import PandasTimeDataset, random_pandasDFClassifier, Predictor
from modules.models import TimeConv1dLSTMNetClassifier, TimeLSTMNetClassifier
from modules.train import TimeTrainer
from torch.optim.lr_scheduler import StepLR
from modules.earlystop import EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    # Parameters
    TIME_STEP = 20
    N_FEATURES = 1
    NUM_CLASSES = 2
    EPOCH = 50

    # Data set up
    train_data_df, val_data_df = random_pandasDFClassifier(batch_size=100, N_FEATURES=N_FEATURES, NUM_CLASSES=NUM_CLASSES)
    train_dataset = PandasTimeDataset(train_data_df, label_col='y', TIME_STEP=TIME_STEP, mode='classifier')
    val_dataset = PandasTimeDataset(val_data_df, label_col='y', TIME_STEP=TIME_STEP, mode='classifier')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model set up
    model = TimeConv1dLSTMNetClassifier(INPUT_DIM=N_FEATURES,
                                        TIME_STEP=TIME_STEP,
                                        HIDDEN_DIM=64,
                                        N_LAYER=2,
                                        OUTPUT_DIM=NUM_CLASSES,
                                        DROPOUT=0.3,
                                        bidirectional=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    early_stopping = EarlyStopping(patience=10, verbose=False)

    # Train set up
    Training = TimeTrainer(train_dataloader=train_dataloader,
                                     val_dataloader=val_dataloader,
                                     model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scheduler=None,
                                     early_stoper=early_stopping,
                                     EPOCH=EPOCH)
    # Run training
    Training.train()

    # Plot loss graph
    Training.plot_loss_graph()

    # Save model
    best_model = True
    Training.save_model(best_model=best_model)

    # Eval
    load_model = TimeConv1dLSTMNetClassifier(INPUT_DIM=N_FEATURES,
                                        TIME_STEP=TIME_STEP,
                                        HIDDEN_DIM=64,
                                        N_LAYER=2,
                                        OUTPUT_DIM=NUM_CLASSES,
                                        DROPOUT=0.3,
                                        bidirectional=False)
    load_model.load_state_dict(torch.load('artifact/best_epoch_model.pt' if best_model == True else 'artifact/last_epoch_model.pt'))
    load_model.eval()

    predictor = Predictor(load_model, train_dataset)
    acc = predictor.confusion_matrix()
    print(acc)

    predictor = Predictor(load_model, val_dataset)
    acc = predictor.confusion_matrix()
    print(acc)




import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import numpy as np
import math
import os
import random
import csv
import json
from tqdm import tqdm
from resolve.utilities import plotting_utils_cnp as plotting
from resolve.utilities import utilities as utils
from resolve.conditional_neural_process import DataGeneration
from resolve.conditional_neural_process import DeterministicModel
from torch.utils.tensorboard import SummaryWriter
import yaml
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

# Set the path to the yaml settings file here
# path_to_settings = "../"

runs = 0
while True:
    runs += 1

    try:
        # Parameters to optimize:
        # Encoder / Decoder: 
        # encoder_sizes = [d_in, hl1, hl2, hl3, hl4, hl5, hl6, hl7, representation_size]
        # decoder_sizes = [representation_size + d_x, hl7, hl6, hl5, hl4, hl3, hl2, hl1, d_out]
        hl1 = random.randint(2, 128)
        hl2 = random.randint(2, 128)
        hl3 = random.randint(2, 128)
        hl4 = random.randint(2, 128)
        hl5 = random.randint(2, 128)
        # hl6 = random.randint(2, 128)
        # hl7 = random.randint(2, 128)
        representation_size = random.randint(2, 128)

        # LEARNING_RATE = 10 ** random.uniform(-5, -3)
        # BATCH_SIZE = random.randint(100, 1000)
        # FILES_PER_BATCH = random.randint(10, 50)
        # TRAINING_EPOCHS = random.randint(1, 3)
        # CONTEXT_IS_SUBSET = bool(random.randint(0, 1))
        # CONTEXT_RATIO = random.uniform(0.1, 0.5)

        # Set the path to the yaml settings file here
        path_to_settings = "/global/cfs/projectdirs/legend/users/tbuerger/neuralNet/binaryBH/resolve/examples/binary-black-hole/"
        with open(f"{path_to_settings}/settings.yaml", "r") as f:
            config_file = yaml.safe_load(f)

        TRAINING_EPOCHS = int(config_file["cnp_settings"]["training_epochs"]) # Total number of training points: training_iterations * batch_size * max_content_points
        torch.manual_seed(0)
        BATCH_SIZE = config_file["cnp_settings"]["batch_size_train"]
        LEARNING_RATE = 0.00001
        FILES_PER_BATCH = config_file["cnp_settings"]["files_per_batch_train"]
        CONTEXT_IS_SUBSET = config_file["cnp_settings"]["context_is_subset"]
        CONTEXT_RATIO = config_file["cnp_settings"]["context_ratio"]

        # Other Parameters
        TEST_AFTER = int(config_file["cnp_settings"]["test_after"])
        torch.manual_seed(0)
        target_range = config_file["simulation_settings"]["target_range"]
        is_binary = target_range[0] >= 0 and target_range[1] <= 1
        version = config_file["path_settings"]["version"]
        path_out = f'{config_file["path_settings"]["path_out_cnp"]}/{version}'
        # Get Features and Labels
        x_size, y_size = utils.get_feature_and_label_size(config_file)


        # CNP Training:
        print("Training Run ", runs)
        d_x, d_in, representation_size, d_out = x_size , x_size+y_size, 32, y_size*2
        encoder_sizes = [d_in, hl1, hl2, hl3, hl4, hl5, representation_size]  # Teste: "[31, 13, 29, 31, 66, 79, 5, 73, 32]"
        decoder_sizes = [representation_size + d_x, hl5, hl4, hl3, hl2, hl1, d_out]

        # encoder_sizes = [d_in] + config_file["cnp_settings"]["encoder_hidden_layers"] + [representation_size]
        # decoder_sizes = [representation_size + d_x]+ config_file["cnp_settings"]["decoder_hidden_layers"] + [d_out] 

        model = DeterministicModel(encoder_sizes, decoder_sizes)
        os.system(f'mkdir -p {path_out}/cnp_{version}_tensorboard_logs/arxiv')
        os.system(f'mv {path_out}/cnp_{version}_tensorboard_logs/events* {path_out}/cnp_{version}_tensorboard_logs/arxiv/')
        writer = SummaryWriter(log_dir=f'{path_out}/cnp_{version}_tensorboard_logs')

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


        USE_DATA_AUGMENTATION = config_file["cnp_settings"]["use_data_augmentation"]
        # load data:
        dataset_train = DataGeneration(mode = "training", 
                                        config_file=config_file, 
                                        path_to_files=config_file["path_settings"]["path_to_files_train"], 
                                        use_data_augmentation=USE_DATA_AUGMENTATION, 
                                        batch_size=BATCH_SIZE,
                                        files_per_batch=FILES_PER_BATCH,
                                        context_ratio=CONTEXT_RATIO)
        dataset_train.set_loader()
        dataloader_train = dataset_train.dataloader

        dataset_test = DataGeneration(mode = "training", 
                                        config_file=config_file, 
                                        path_to_files=config_file["path_settings"]["path_to_files_testing"], 
                                        use_data_augmentation=False, 
                                        batch_size=BATCH_SIZE,
                                        files_per_batch=FILES_PER_BATCH,
                                        context_ratio=CONTEXT_RATIO)
        dataset_test.set_loader()
        dataloader_test = dataset_test.dataloader


        bce = nn.BCELoss()

        for it_epoch in range(TRAINING_EPOCHS):
            data_iter = iter(dataloader_test)
            for b, batch in tqdm(enumerate(dataloader_train), total=math.ceil(len(dataloader_train)/BATCH_SIZE), desc="Training Epoch {}/{}".format(it_epoch+1, TRAINING_EPOCHS)):
                it_step = it_epoch * len(dataloader_train) + b
                batch_formated=dataset_train.format_batch_for_cnp(batch, CONTEXT_IS_SUBSET)
                # Get the predicted mean and variance at the target points for the testing set
                log_prob, mu, _ = model(batch_formated.query, batch_formated.target_y, is_binary)
                
                # Define the loss
                loss = -log_prob.mean()
                loss.backward()

                # Perform gradient descent to update parameters
                optimizer.step()
            
                # reset gradient to 0 on all parameters
                optimizer.zero_grad()
                
                if is_binary:
                    loss_bce = bce(mu, batch_formated.target_y)
                else:
                    loss_bce=-1
                
                # Inside your batch loop, right after computing losses:
                writer.add_scalar('Loss/Logprob/train', loss.item(), it_step)
                y_pred = mu[0].detach().cpu().numpy().flatten()
                y_true = batch_formated.target_y[0].detach().cpu().numpy().flatten()
                mae = mean_absolute_error(y_true,y_pred)
                mse = mean_squared_error(y_true,y_pred)
                r2 = r2_score(y_true, y_pred)
                writer.add_scalar('Metric/Mae/Test', mae, it_step)
                writer.add_scalar('Metric/Mse/Test', mse, it_step)
                writer.add_scalar('Metric/R2/Test', r2, it_step)
                if is_binary:
                    writer.add_scalar('Loss/BCE/train', loss_bce.item(), it_step)
                    y_pred = (y_pred > 0.5).astype(int)
                    y_true = y_true.astype(int)
                    acc = accuracy_score(y_true, y_pred)
                    writer.add_scalar('Accuracy/train', acc, global_step=it_step)
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    writer.add_scalar('Metrics/Precision/Train', precision, global_step=it_step)
                    writer.add_scalar('Metrics/Recall/Train', recall, global_step=it_step)
                    writer.add_scalar('Metrics/F1/Train', f1, global_step=it_step)
                
                mu=mu[0].detach().numpy()
                if b % TEST_AFTER == 0:
                    try:
                        batch_testing = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader_test)
                        batch_testing = next(data_iter)

                    batch_formated_test=dataset_test.format_batch_for_cnp(batch_testing, CONTEXT_IS_SUBSET)
                
                    log_prob_testing, mu_testing, _ = model(batch_formated_test.query, batch_formated_test.target_y, is_binary)
                    loss_testing = -log_prob_testing.mean()
                    
                    if is_binary:
                        loss_bce_testing = bce(mu_testing,  batch_formated_test.target_y)
                    else:
                        loss_bce_testing = -1.

                    writer.add_scalar('Loss/Logprob/Test', loss_testing.item(), b+it_epoch * len(dataloader_train))
                    if is_binary:
                        writer.add_scalar('Loss/BCE/Test', loss_bce_testing.item(),b+it_epoch * len(dataloader_train))
                        y_pred = (mu_testing[0].detach().cpu().numpy() > 0.5).astype(int).flatten()
                        y_true = batch_formated_test.target_y[0].detach().cpu().numpy().astype(int).flatten()
                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                        TN, FP, FN, TP = cm.ravel()
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted Label')
                        ax.set_ylabel('True Label')
                        ax.set_title(f'Confusion Matrix at step {it_step}')
                        writer.add_figure('ConfusionMatrix', fig, global_step=it_step)
                        writer.add_scalar('ConfusionMatrix/FalsePositives/Test', cm[0, 1], it_step)
                        writer.add_scalar('ConfusionMatrix/FalseNegatives/Test', cm[1, 0], it_step)
                        # Predicted probabilities
                        acc = accuracy_score(y_true, y_pred)
                        writer.add_scalar('Accuracy/Test', acc, global_step=it_step)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        writer.add_scalar('Metrics/Precision/Train', precision, global_step=it_step)
                        writer.add_scalar('Metrics/Recall/Train', recall, global_step=it_step)
                        writer.add_scalar('Metrics/F1/Train', f1, global_step=it_step)

                    mu_testing = mu_testing[0].detach().numpy()

                    y_true = batch_formated_test.target_y[0].detach().numpy()
                    mae = mean_absolute_error(y_true,mu_testing)
                    mse = mean_squared_error(y_true,mu_testing)
                    r2 = r2_score(y_true, y_pred)
                    writer.add_scalar('Metric/Mae/Test', mae, it_step)
                    writer.add_scalar('Metric/Mse/Test', mse, it_step)
                    writer.add_scalar('Metric/R2/Test', r2, it_step)

                    if y_size ==1:
                        fig = plotting.plot(mu, batch_formated.target_y[0].detach().numpy(), f'{loss:.2f}', mu_testing, batch_formated_test.target_y[0].detach().numpy(), f'{loss_testing:.2f}', target_range, it_step)
                        writer.add_figure('Prediction/train_vs_test', fig, global_step=it_step)
                    else:
                        for k in range(y_size):
                            fig = plotting.plot(mu[:,k], batch_formated.target_y[0].detach().numpy()[:,k], f'{loss:.2f}', mu_testing[:,k], batch_formated_test.target_y[0].detach().numpy()[:,k], f'{loss_testing:.2f}', target_range, it_step)
                            writer.add_figure(f'Prediction/train_vs_test_k{k}', fig, global_step=it_step)



        writer.close()
        torch.save(model.state_dict(), f'{path_out}/cnp_{version}_model.pth')

        config_file["feature_settings"]["x_mean"] = dataset_train.feature_mean.numpy().tolist()
        config_file["feature_settings"]["x_std"] = dataset_train.feature_std.numpy().tolist()
        # Save back to the file
        with open(f"{path_to_settings}/settings.yaml", "w") as f:
            yaml.safe_dump(config_file, f, sort_keys=False)


        # Summary to csv
        # Prepare the final row dictionary
        row = {
            # Hyperparameters
            "Encoder Sizes": encoder_sizes,
            "Decoder Sizes": decoder_sizes,
            "Representation Size": representation_size,
            "Learning Rate": LEARNING_RATE,
            "Batch Size": BATCH_SIZE,
            "Files per Batch": FILES_PER_BATCH,
            "Training Epochs": TRAINING_EPOCHS,
            "Context Ratio": CONTEXT_RATIO,
            "Binary Task": is_binary,
            "Data Augmentation": config_file["cnp_settings"]["use_data_augmentation"],
            "Context is Subset": CONTEXT_IS_SUBSET,
            "Output Path": path_out,
            "Model Checkpoint": f'{path_out}/cnp_{version}_model.pth',
            "TensorBoard Logs": f'{path_out}/cnp_{version}_tensorboard_logs',

            # Final Metrics
            "Final Train LogProb Loss": loss.item(),
            "Final Train BCE Loss": loss_bce.item() if is_binary else None,
            "Final Test LogProb Loss": loss_testing.item(),
            "Final Test BCE Loss": loss_bce_testing.item() if is_binary else None,
            "Final Accuracy Test": acc,
            "Final Precision": precision,
            "Final Recall": recall,
            "Final F1": f1,
            "Final MAE": mae,
            "Final MSE": mse,
            "Final R2": r2,

            # Confusion Matrix Elements (only meaningful for binary task)
            "True Positives": TP if is_binary else None,
            "False Positives": FP if is_binary else None,
            "True Negatives": TN if is_binary else None,
            "False Negatives": FN if is_binary else None,
        }

        csv_path = f"{path_out}/cnp_training_summary.csv"
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"Run {runs} in Summary csv geschrieben")

    except Exception as e:
        print(f"Fehler: {e}")

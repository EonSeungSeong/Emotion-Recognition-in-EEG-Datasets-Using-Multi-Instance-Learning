import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
import os
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import wandb
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, lfilter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Use one CPU core
torch.set_num_threads(1)

# Set random seed for reproducibility
def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    Args:
        seed (int): Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GPU ID Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# Bandpass filter
def bandpass_filter(data, lowcut=0.16, highcut=43, fs=128, order=5):
    """
    Apply bandpass filter to the EEG data.

    Args:
        data (numpy.ndarray): EEG data of shape [num_channels, num_samples].
        lowcut (float): Low frequency cutoff (Hz).
        highcut (float): High frequency cutoff (Hz).
        fs (int): Sampling frequency (Hz).
        order (int): Order of the filter.

    Returns:
        filtered_data (numpy.ndarray): Filtered EEG data of the same shape as input.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalize lowcut frequency
    high = highcut / nyquist  # Normalize highcut frequency

    # Ensure Wn is within the valid range
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1")

    # Design bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to each channel
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[0]):
        filtered_data[channel] = lfilter(b, a, data[channel])
    
    return filtered_data

# Custom Dataset
class MILDataset(Dataset):
    def __init__(self, data, labels, instance_size=128):
        """
        Args:
            data (list of numpy.ndarray): List of EEG data (each of shape [num_channels, num_samples])
            labels (list of int): List of labels corresponding to each EEG data
            instance_size (int): Number of samples per instance
        """
        self.data = data
        self.labels = labels
        self.instance_size = instance_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]  # [num_channels, num_samples]
        label = self.labels[idx]

        # Split data into instances
        num_instances = data.shape[1] // self.instance_size
        instances = data[:, :num_instances * self.instance_size]
        instances = instances.reshape(num_instances, data.shape[0], self.instance_size)  # [num_instances, num_channels, instance_size]

        return torch.tensor(instances, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
# Load and preprocess data
def load_eeg_data(base_path, participants, sampling_rate=128, normalization=None):
    """
    Load EEG data from CSV files, apply sampling rate, normalization, and group by participants.

    Args:
        base_path (str): Path to the base directory containing EEG data.
        participants (list): List of participant folder names.
        sampling_rate (int): Sampling rate to downsample the data (default: 128).
        normalization (str): Normalization method ('minmax', '[-1,1]', 'standard', or None).

    Returns:
        data_by_participant (list): List of EEG data arrays grouped by participant.
        labels_by_participant (list): List of labels grouped by participant.
    """
    data_by_participant = []
    labels_by_participant = []

    for participant in participants:
        participant_path = os.path.join(base_path, participant, "Preprocessed EEG Data", ".csv format")
        participant_data = []
        participant_labels = []

        for csv_file in os.listdir(participant_path):
            if csv_file.endswith(".csv"):
                file_path = os.path.join(participant_path, csv_file)
                # Load CSV file
                df = pd.read_csv(file_path)
                # Remove Unnamed column
                df = df.drop(columns=["Unnamed: 14"], errors='ignore')
                # Apply sampling rate
                data = df.to_numpy(dtype=np.float32)[::sampling_rate].T  # Transpose to [num_channels, num_samples]
                # Apply bandpass filter

                # **앞뒤로 38개 샘플 제거**
                if data.shape[1] > 76:  # Ensure enough samples are available
                    data = data[:, 38:-38]  # Remove first and last 38 samples
                else:
                    print(f"Warning: File {csv_file} has insufficient samples after removing 38 from both sides.")
                    continue  # Skip this file
                data = bandpass_filter(data, lowcut=0.16, highcut=43, fs=128, order=5)
                # Normalize data
                if normalization == 'minmax':
                    scaler = MinMaxScaler()
                    data = scaler.fit_transform(data.T).T  # Normalize per feature
                elif normalization == '[-1,1]':
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    data = scaler.fit_transform(data.T).T
                elif normalization == 'standard':
                    scaler = StandardScaler()
                    data = scaler.fit_transform(data.T).T

                label = int(csv_file.split("G")[1][0])  # Extract label from filename (G1, G2, G3, G4 -> 1, 2, 3, 4)
        
                label -= 1  # Change labels to start from 0 (0, 1, 2, 3)
                participant_data.append(data)  # Append the entire normalized data
                participant_labels.append(label)  # Append the label

        if participant_data:
            data_by_participant.append(participant_data)
            labels_by_participant.append(participant_labels)

    return data_by_participant, labels_by_participant

def visualize_instance_attention(model, val_loader, fold, class_names=None, save_dir="./attention_visualizations"):
    """
    Visualize instance-level attention in a MIL model using heatmap-style visualization.
    
    Args:
        model (nn.Module): Trained MIL model with attention mechanism.
        val_loader (DataLoader): Validation DataLoader.
        fold (int): Current fold index for saving images.
        class_names (list of str): List of class names for labels.
        save_dir (str): Directory to save the visualizations.
    """
    model.eval()
    save_path = os.path.join(save_dir, f"fold_{fold + 1}")
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (instances, labels) in enumerate(val_loader):
            instances, labels = instances.cuda(), labels.cuda()
            batch_size, num_instances, num_channels, instance_size = instances.size()

            # Extract features and attention weights
            features = model.instance_extractor(instances.view(-1, num_channels, instance_size))  # [batch_size * num_instances, 64, 1]
            features = features.view(batch_size, num_instances, -1)  # [batch_size, num_instances, 64]
            attention_weights = model.attention(features).squeeze(-1)  # [batch_size, num_instances]
            attention_weights = F.softmax(attention_weights, dim=1).cpu().numpy()  # Normalize weights
            
            for i in range(batch_size):
                label = labels[i].item()
                attention = attention_weights[i]  # Attention weights for instances
                
                # Log scale transformation to emphasize differences
                attention_log_scaled = np.log1p(attention * 100)  # Scale and apply log transform
                
                # Heatmap preparation
                heatmap = np.expand_dims(attention_log_scaled, axis=0)  # [1, num_instances]
                heatmap = np.repeat(heatmap, 10, axis=0)  # Stretch vertically for better visualization

                # Plot heatmap
                plt.figure(figsize=(12, 2))  # Adjust figure size
                plt.imshow(heatmap, aspect="auto", cmap="hot", origin="lower")
                plt.title(f"Fold {fold + 1} - Bag {batch_idx + 1} - Class {class_names[label] if class_names else label}")
                plt.xlabel("Instance Index")
                plt.tight_layout()

                # Save visualization
                plt.savefig(os.path.join(save_path, f"bag_{batch_idx + 1}_class_{label}_attention_heatmap.png"))
                plt.close()

# This is an unofficial custom model that modifies the structure of DeepMIL....
class DeepMILModel(nn.Module):
    def __init__(self, num_channels=14, instance_size=128, num_classes=4):
        super(DeepMILModel, self).__init__()
        # Instance-level feature extractor
        self.instance_extractor = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling over the temporal dimension
        )

        # Attention-based aggregation
        self.attention = nn.Sequential(
            nn.Linear(64, 32),  # Reduce feature dimensions
            nn.Tanh(),
            nn.Linear(32, 1)    # Output attention weights
        )

        # Bag-level classifier
        self.classifier = nn.Linear(64, num_classes)  # Classifier head for final predictions

    def forward(self, instances):
        """
        Args:       
            instances: Tensor of shape [batch_size, num_instances, num_channels, instance_size]
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        # Ensure the input shape is valid
        if len(instances.size()) != 4:
            raise ValueError("Expected input of shape [batch_size, num_instances, num_channels, instance_size]")

        batch_size, num_instances, num_channels, instance_size = instances.size()

        # Reshape for instance-level feature extraction
        instances = instances.view(-1, num_channels, instance_size)  # Flatten to [batch_size * num_instances, num_channels, instance_size]
        features = self.instance_extractor(instances).squeeze(-1)    # Extracted features: [batch_size * num_instances, 64]

        # Reshape back to [batch_size, num_instances, 64]
        features = features.view(batch_size, num_instances, -1)

        # Compute attention weights
        attention_weights = self.attention(features).squeeze(-1)    # [batch_size, num_instances]
        attention_weights = F.softmax(attention_weights, dim=1)     # Normalize attention scores

        # Compute bag-level features using attention
        bag_features = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)  # [batch_size, 64]

        # Classify the bag-level features
        logits = self.classifier(bag_features)  # [batch_size, num_classes]

        return logits


# Train and evaluate the model
seed =42
def train_and_evaluate(data_by_participant, labels_by_participant, num_epochs=1000, k_folds=10, batch_size=64, instance_size=128, patience=10):
    set_seed(seed)  # Set random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # WandB 초기화
    wandb.init(project="EEG_Classification", name="MILModel_Training_instance128", config={
        "num_epochs": num_epochs,
        "k_folds": k_folds,
        "batch_size": batch_size,
        "instance_size": instance_size,
        "learning_rate": 0.005,
        "patience": patience
    })
    config = wandb.config
    participants = list(range(len(data_by_participant)))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_accuracies = []  # To store accuracy for each fold
    fold_f1 = []
    fold_precision = []
    fold_recall = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(participants)):
        print(f"\nFold {fold + 1}/{k_folds}")

        # Prepare train and validation data
        train_data = [data_by_participant[i] for i in train_idx]
        train_labels = [labels_by_participant[i] for i in train_idx]
        val_data = [data_by_participant[i] for i in val_idx]
        val_labels = [labels_by_participant[i] for i in val_idx]

        train_data = [item for sublist in train_data for item in sublist]  # Flatten list
        train_labels = [item for sublist in train_labels for item in sublist]
        val_data = [item for sublist in val_data for item in sublist]
        val_labels = [item for sublist in val_labels for item in sublist]

        train_dataset = MILDataset(train_data, train_labels, instance_size)
        val_dataset = MILDataset(val_data, val_labels, instance_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Initialize model, optimizer, and loss function
        #model = AdvancedMILModel().to(device)
        model = DeepMILModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Track best validation loss and early stopping
        best_val_loss = float('inf')
        best_val_accuracy = float('-inf')
        early_stop_counter = 0  # Counter for early stopping

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for instances, labels in train_loader:
                instances, labels = instances.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(instances)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # Validation loop
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for instances, labels in val_loader:
                    instances, labels = instances.to(device), labels.to(device)
                    outputs = model(instances)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    #print(outputs)
                    #print(outputs.size())
                    _, predicted = torch.max(outputs, 1) #
                    #print(predicted)
                    #print(labels)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            
            # WandB Logging
            wandb.log({
                f"Fold {fold + 1} Train Loss": train_loss,
                f"Fold {fold + 1} Val Loss": val_loss,
                f"Fold {fold + 1} Val Accuracy": val_accuracy
            })

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #best_model_state = model.state_dict()  # Save best model state
                early_stop_counter = 0  # Reset counter
                print(f"Saving best model with validation loss: {best_val_loss:.4f}")
                torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pth")
            else:
                early_stop_counter += 1  # Increment counter

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}%")

            # Early stopping
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save best model for the fold
        print(f"Best Validation Loss for Fold {fold + 1}: {best_val_loss:.4f}")

        # Load best model
        model.load_state_dict(torch.load(f"best_model_fold_{fold + 1}.pth"))  # Load from file

        model.eval()
        class_names = ["G1", "G2", "G3", "G4"]
        # Generate XAI visualizations for the fold
        #Generate CAM-like visualizations for the fold
        visualize_instance_attention(
            model=model,
            val_loader=val_loader,
            fold=fold,
            class_names=["G1", "G2", "G3", "G4"],
            save_dir="./cam_visualizations"
        )
        # Evaluate best model on validation set
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for instances, labels in val_loader:
                instances, labels = instances.to(device), labels.to(device)
                outputs = model(instances)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"Confusion Matrix for Fold {fold + 1}:\n{cm}")
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(4))).plot()
        plt.title(f"Confusion Matrix for Fold {fold + 1}")
        # 이미지 저장
        plt.savefig(f"confusion_matrix_fold_{fold + 1}.png")

        # Final Accuracy for Fold
        fold_accuracy = 100 * correct / total
        fold_accuracies.append(fold_accuracy)
        f1 = f1_score(all_labels, all_predictions, average="weighted")
        precision = precision_score(all_labels, all_predictions, average="weighted")
        recall = recall_score(all_labels, all_predictions, average="weighted")        
        fold_f1.append(f1)
        fold_precision.append(precision)
        fold_recall.append(recall)
        print(f"Fold {fold + 1} - Accuracy: {fold_accuracy:.2f}%, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Sensitivity(Recall): {recall:.4f}")
        wandb.log({
            f"Fold {fold + 1} Accuracy": fold_accuracy,
            f"Fold {fold + 1} F1-Score": f1,
            f"Fold {fold + 1} Precision": precision,
            f"Fold {fold + 1} Recall": recall
        })
    # Overall Accuracy
    print(fold_accuracies)
    print(fold_f1)
    print(fold_precision)
    print(fold_recall)
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    avg_f1 = sum(fold_f1) / len(fold_f1)
    avg_precision = sum(fold_precision) / len(fold_precision)
    avg_recall = sum(fold_recall) / len(fold_recall)
    print(f"\nOverall Accuracy across {k_folds} folds: {avg_accuracy:.4f}%")
    print(f"Overall F1-Score across {k_folds} folds: {avg_f1:.4f}")
    print(f"Overall Precision across {k_folds} folds: {avg_precision:.4f}")
    print(f"Overall Recall across {k_folds} folds: {avg_recall:.4f}")
    wandb.log({"Overall Accuracy": avg_accuracy})
    wandb.log({"Overall F1-Score": avg_f1})
    wandb.log({"Overall Precision": avg_precision})
    wandb.log({"Overall Recall": avg_recall})


# Base path to data
base_path = "/workspace/data/eeg_project/GAMEEMO"
participants = [f"(S{str(i).zfill(2)})" for i in range(1, 29)]  # S01 to S28

# Choose normalization method
normalization_method = 'standard'  # Options: 'minmax', '[-1,1]', 'standard', None

# Load data by participant with sampling rate and normalization
sampling_rate = 1 # 1 means no downsampling
data_by_participant, labels_by_participant = load_eeg_data(
    base_path, participants, sampling_rate=sampling_rate, normalization=normalization_method
)
instance_size = 128  # 128 - 1 second, 256 - 2 seconds, 512 - 4 seconds ....
# Call the training function
train_and_evaluate(data_by_participant, labels_by_participant, instance_size=instance_size,patience=50,k_folds=10)

wandb.finish()
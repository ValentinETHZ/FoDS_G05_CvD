import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Data loading and preprocessing
# -------------------------
data = pd.read_csv(
    filepath_or_buffer="../data/2025_cardio_train.csv",
    index_col=0,
    na_filter=False,
    dtype={
        "age": "int",
        "height": "int",
        "weight": "float",
        "gender": "category",
        "ap_hi": "int",
        "ap_lo": "int",
        "cholesterol": "category",
        "gluc": "category",
        "smoke": "category",
        "alco": "category",
        "active": "category",
        "cardio": "category",
    }
)

# Filtering the data for measuring errors
data = data[data["ap_hi"] >= 0]
data = data[data["ap_hi"] <= 200]
data = data[data["ap_lo"] >= 0]
data = data[data["ap_lo"] <= 200]

# Order ordinal columns
data["cholesterol"] = data["cholesterol"].cat.as_ordered()
data["gluc"] = data["gluc"].cat.as_ordered()

# One-hot encoding of categorical data
cat_cols = data.select_dtypes(include="category").columns
for col in cat_cols:
    data = pd.get_dummies(data, columns=[col], drop_first=True, dtype=int)

print(data.tail(5))

# Construct features and labels
X = data.drop('cardio_1', axis=1)
y = data['cardio_1']

# Split into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2025)

# Apply data preprocessing; scaling numerical columns
sc = StandardScaler(with_mean=False, with_std=False)
num_cols = X.select_dtypes(include=['int', 'float']).columns

X_train[num_cols] = sc.fit_transform(X_train[num_cols])
X_test[num_cols] = sc.transform(X_test[num_cols])

# Convert pandas DataFrame to numpy array
X_train, X_test, y_train, y_test = (
    np.array(X_train),
    np.array(X_test),
    np.array(y_train),
    np.array(y_test),
)


# -------------------------
# Define the PyTorch Dataset
# -------------------------
class CardioDataset(Dataset):
    def __init__(self, features, labels):
        # Convert data to PyTorch tensors
        self.X = torch.tensor(features, dtype=torch.float32)
        # For classification tasks in PyTorch, labels are usually LongTensor
        self.y = torch.tensor(labels.astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CardioDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -------------------------
# Dataframe setup Tuning
# -------------------------
columns = ['Epoch', 'Learning Rate', 'Layers', 'Accuracy']
df = pd.DataFrame(columns=columns, dtype=float)

# -------------------------
# Hyperparameter Tuning
# -------------------------
rng_layers = np.linspace(1,10,10)     # Range of hidden layers
rng_learning_rate = np.logspace(-5, -3, 8)       # Range of learning rate
rng_epoch = np.logspace(0, 2, 5)      # Range of number of epochs

print(f'Model will be evaluated: Layers: {rng_layers}, Learning rate: {rng_learning_rate}, Epoch: {rng_epoch}')

# Best accuracy setup
best_acc = 0
best_params = 0, 0, 0

# looping over every model variant
for n_epochs in rng_epoch:
    for n_layers in rng_layers:
        for lr in rng_learning_rate:
            # -------------------------
            # Define a simple Deep Neural Network model
            # -------------------------
            class CardioDNN(nn.Module):
                def __init__(self, input_size, hidden_units, num_layers, num_classes, activation_fn=nn.ReLU):
                    """
                    Args:
                        input_size (int): The number of features in the input.
                        hidden_units (int): The size (number of neurons) in each hidden layer.
                        num_layers (int): Number of hidden layers.
                        num_classes (int): Number of outputs.
                        activation_fn: The activation function to use; default is ReLU.
                    """
                    super(CardioDNN, self).__init__()

                    # Create a ModuleList for hidden layers
                    self.hidden_layers = nn.ModuleList()

                    # First layer: from input size to the first hidden layer
                    self.hidden_layers.append(nn.Linear(input_size, hidden_units))

                    # For the subsequent layers, from hidden_units -> hidden_units
                    for _ in range(num_layers - 1):
                        self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))

                    # Output layer from hidden_units to num_classes
                    self.output_layer = nn.Linear(hidden_units, num_classes)
                    self.activation = activation_fn()

                def forward(self, x):
                    # Pass input through each hidden layer followed by activation
                    for layer in self.hidden_layers:
                        x = self.activation(layer(x))
                    # Final linear output without activation (logits)
                    x = self.output_layer(x)
                    return x



            # Example instantiation:
            input_size = X_train.shape[1]           # number of features from your dataset
            num_classes = len(np.unique(y_train))   # e.g., 2 for binary classification
            hidden_units = 64                       # you can adjust this parameter
            num_layers = int(n_layers)              # for example, creating 10 hidden layers

            model = CardioDNN(input_size, hidden_units, num_layers, num_classes)


            # -------------------------
            # Loss and Optimizer
            # -------------------------
            # Use CrossEntropyLoss for classification
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)


            # -------------------------
            # Training Loop
            # -------------------------
            num_epochs = int(n_epochs)

            for epoch in range(num_epochs):
                running_loss = 0.0
                model.train()  # Ensure the model is in training mode
                for inputs, labels in train_loader:
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

            print("Training complete!")


            # -------------------------
            # Evaluation
            # -------------------------
            # Ensure the model is in evaluation mode
            model.eval()

            # Convert X_test to a torch tensor (if not already) and ensure it is the correct type
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

            # Disable gradient calculations; we aren't training here
            with torch.no_grad():
                # Get model predictions (logits are output by the network)
                y_pred_logits = model(X_test_tensor)
                # For classification problems with CrossEntropyLoss,
                # the predicted class is the one with the highest logit.
                y_pred = torch.argmax(y_pred_logits, dim=1).numpy()

            # Convert y_test to a NumPy array if it isn't already
            y_true = np.array(y_test)

            # Compute classification accuracy
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy > best_acc:
                best_acc = accuracy
                print("New best accuracy so far!")
                best_params = lr, lr, int(n_layers), int(n_epochs)

            new_row = {'Epoch': n_epochs, 'Learning Rate': lr, 'Layers': n_layers, 'Accuracy': accuracy}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Print out the metrics
            print("Accuracy: {:.4f} with LR: {:.5f}, Number of Layers: {}, Number of Epochs: {}, ".format(accuracy, lr, int(n_layers), int(n_epochs)))

# best parameters output
print(f'The best accuracy of {best_acc} was achieved with the following parameters: {best_params}')

# Create the 'data' folder if it doesn't exist
folder = "values"
os.makedirs(folder, exist_ok=True)

# Format today's date for the filename
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{time_stamp}.csv"
path = os.path.join(folder, filename)

# Save the DataFrame as a CSV file
df.to_csv(path)

# -------------------------
# Hyperparameter tuning visualisation
# -------------------------

# define pairs to be visualised
pairs = [
    ('Layers', 'Epoch', 0.001),
    ('Layers', 'Learning Rate', 10),
    ('Epoch', 'Learning Rate', 5)
]


for x_col, y_col, val in pairs:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # filter the dataframe for suggested values
    for column in data.columns:
        if column not in pairs:
            data_clone = data[data[column] == val]

    x = data_clone[x_col]
    y = data_clone[y_col]
    z = data_clone['Accuracy']

    # 1a. As points:
    ax.scatter(
        x, y, z,
        c=data_clone['Accuracy'],
        cmap='viridis',
        marker='o',
        s=20,
        alpha=0.8
    )

    # 1b. Or as a triangular mesh surface:
    tri = Triangulation(x, y)
    ax.plot_trisurf(
        tri, z,
        cmap='viridis',
        edgecolor='none',
        alpha=0.9
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel('Accuracy')
    ax.set_title('Triangulated Surface / Scatter of scattered data')

    plt.show()

    save_dir = "visualisation"
    save_path = os.path.join(save_dir, f"{x_col}_{y_col}.png")
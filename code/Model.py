import pandas as pd
import numpy as np
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
num_layers = int(9)              # for example, creating 10 hidden layers

model = CardioDNN(input_size, hidden_units, num_layers, num_classes)


# -------------------------
# Loss and Optimizer
# -------------------------
# Use CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005179)


# -------------------------
# Training Loop
# -------------------------
num_epochs = int(100)

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

# Print out the metrics
print("Accuracy: {:.4f}".format(accuracy))



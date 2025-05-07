import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Function for DNN
# -------------------------

def DNN_func(X_train, X_test, y_train, y_test) -> ndarray:

    ### Data loading and conversion ###

    # Convert pandas DataFrame to numpy array
    X_train, X_test, y_train, y_test = (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test),
    )

    ### Define the PyTorch Dataset ###

    # Setup PyTorch tensor class
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

    ### Define a simple Deep Neural Network model

    # Setup PyTorch DNN class
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
    input_size = X_train.shape[1]  # number of features from your dataset
    num_classes = len(np.unique(y_train))  # e.g., 2 for binary classification
    hidden_units = 64  # you can adjust this parameter
    num_layers = int(9)  # for example, creating 9 hidden layers

    model = CardioDNN(input_size, hidden_units, num_layers, num_classes)

    ### Loss and Optimizer ###

    # Use CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005179)

    ### Training Loop ###

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

    ### Evaluation ###

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

    return y_pred
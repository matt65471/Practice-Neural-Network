import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int64)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, stratify=y
)

X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Declares a neural network with 3 layers. The number of neurons goes from
# 128 --> 64 --> 10. ReLU is applied at each step to add nonlinearity to learn
# more complex patterns. Linear here is z = wx + b. The complete process is 
# 784 -> 128 -> ReLU -> 64 -> ReLU -> 10.
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # Returns output after passing in an input of x
        return self.model(x)

model = DigitNet()

# Declare Cross Entropy Loss, but no need for including softmax because 
# it is implemented here already. Softmax helps with stability by making the 
# numbers smaller.
criterion = nn.CrossEntropyLoss()

# Declare gradient descent
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        # Get result of the training
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Use chain rule to calculate gradient for each step
        loss.backward()
        # Goes back and adjusts everything with calculated gradient
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Start evaluating the NN that we created
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")




# Try out NN by drawing

grid = np.zeros((28, 28), dtype=int)

drawing = False

fig, ax = plt.subplots()
img = ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)

# Show grid lines
ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

def update_cell(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    x = int(event.xdata)
    y = int(event.ydata)
    if 0 <= x < 28 and 0 <= y < 28:
        grid[y, x] = 1
        img.set_data(grid)
        fig.canvas.draw_idle()

def on_press(event):
    global drawing
    if event.button == 1:
        drawing = True
        update_cell(event)

def on_release(event):
    global drawing
    drawing = False

def on_move(event):
    if drawing:
        update_cell(event)

def on_key(event):
    global grid
    if event.key == "c":
        grid[:] = 0
        img.set_data(grid)
        fig.canvas.draw_idle()
    elif event.key == "r":
        flat_grid = grid.flatten()
        input_tensor = torch.tensor(flat_grid, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            plt.title("Draw on 28x28 Grid\nLeft click + drag to draw | c = clear | r = predict digit" + f" | Predicted: {predicted.item()}")
            plt.draw()

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.title("Draw on 28x28 Grid\nLeft click + drag to draw | c = clear | r = predict digit")
plt.show()
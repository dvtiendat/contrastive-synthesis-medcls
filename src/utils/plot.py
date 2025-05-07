import matplotlib.pyplot as plt

def plot_loss(train_loss):
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("SimCLR Training Loss")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def save_loss_history(train_loss, test_loss, epochs):
    total_epochs = np.linspace(0, epochs-1, epochs)
    fig, ax = plt.subplots(figsize=(12,8))
    plt.title('Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.plot(total_epochs, np.array(train_loss), label='Train Loss')
    ax.plot(total_epochs, np.array(test_loss), label='Val Loss')
    ax.legend(prop={'size': 15})
    # plt.show()
    plt.savefig('output/loss.png')

def save_accuracy_history(train_acc, test_acc, epochs):
    total_epochs = np.linspace(0, epochs-1, epochs)
    fig, ax = plt.subplots(figsize=(12,8))
    plt.title('Accuracy', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.plot(total_epochs, np.array(train_acc), label='Train Accuracy')
    ax.plot(total_epochs, np.array(test_acc), label='Val Accuracy')
    ax.legend(prop={'size': 15})
    # plt.show()
    plt.savefig('output/acc.png')

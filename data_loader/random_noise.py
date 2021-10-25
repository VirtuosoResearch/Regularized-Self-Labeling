import os
import torch
import numpy as np
from scipy.special import softmax

def label_noise(dataset, train_indices, noise_rate, symmetric=True):
    assert 0 <= noise_rate <= 1
    # Fix seed to flip the labels
    np.random.seed(1024)
    
    # setup
    num_classes = np.max(dataset.labels)+1
    train_labels = np.asarray([dataset.labels[i] for i in train_indices])
    train_labels_old = np.copy(train_labels)

    # randomize labels with given noise rate
    n_train = len(train_labels)
    n_rand = int(noise_rate * n_train)
    randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
    if symmetric:
        train_labels[randomize_indices] = np.random.choice(range(num_classes), size=n_rand, replace=True)
    else:
        probs = np.random.rand(num_classes)
        probs = softmax(probs)
        print(probs)
        train_labels[randomize_indices] = np.random.choice(range(num_classes), size=n_rand, replace=True, p=probs)

    wrong_indices = np.where(train_labels != train_labels_old)[0]

	# apply the change to original dataset
    for i, index in enumerate(train_indices):
        dataset.labels[index] = train_labels[i]

    return wrong_indices, train_labels_old

def noisy_labeler(train_data_loader, dataset, train_indices, aux_model, device):
    """ use a file to saved noisy labels """
    torch.manual_seed(1024)
    # setup
    all_labels = np.asarray(dataset.labels)
    train_labels_old = np.asarray([dataset.labels[i] for i in train_indices])

    # generate labels with by the prediction of the auxiliary model
    aux_model.to(device)
    torch.no_grad()
    for idx, (data, target, index) in enumerate(train_data_loader):
        data, target = data.to(device), target.to(device)
        outputs = aux_model(data)
        _, pred_labels = torch.exp(outputs).max(dim=1)
        
        randomize_indices = index.numpy()
        pred_labels = pred_labels.to("cpu").numpy()
        all_labels[randomize_indices] = pred_labels

    train_labels = np.asarray([all_labels[i] for i in train_indices])
    wrong_indices = np.where(train_labels != train_labels_old)[0]

	# apply the change to original dataset
    for idx in train_indices:
        dataset.labels[idx] = all_labels[idx]

    return wrong_indices, train_labels_old
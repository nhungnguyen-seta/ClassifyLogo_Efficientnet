import torch
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import shutil
import time
from datasets import get_datasets, get_data_loaders
matplotlib.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion, checkpoint_name):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"./output/"+str(checkpoint_name))

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./output/accuracy_100epoch_17112022.png")
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./output/loss_100epoch_17112022.png")

def split_train_test(data_dir='./Dataset', TRAIN_RATIO = 0.8):
    train_dir = './train'
    test_dir = './test'

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    classes = os.listdir(data_dir)
    print(classes)
    for c in classes:
        
        class_dir = os.path.join(data_dir, c)
        
        images = os.listdir(class_dir)
        
        n_train = int(len(images) * TRAIN_RATIO)
        
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        os.makedirs(os.path.join(train_dir, c), exist_ok = True)
        os.makedirs(os.path.join(test_dir, c), exist_ok = True)
        
        count = 1
        for image in train_images:
            image_name = 'logo_'+c+str(count)+'.png'
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image_name) 
            shutil.copyfile(image_src, image_dst)
            count=count+1
        count = 1
        for image in test_images:
            image_name = 'logo_'+c+str(count)+'.png'
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image_name) 
            shutil.copyfile(image_src, image_dst)
            count=count+1
if __name__ == '__main__':

    os.chdir(os.getcwd()+'/nhung/Efficientnet')
    split_train_test()
    #dataset_train, dataset_valid, dataset_classes = get_datasets()
    #train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
    #f = open("class_names.py", "w")
    #f.write("class_names = "+str(train_loader.dataset.classes))
    #f.close()

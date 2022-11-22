import numpy as np
import cv2
import torch
import glob as glob
import os
import time

from torchvision import transforms
from torch.nn import functional as F
from torch import topk

from model import build_model
from class_names import class_names
import time


os.makedirs('output/results/', exist_ok=True)

# Define computation device.
device = 'cpu'
# Class names.
# Initialize model, switch to eval model, load trained weights.
model = build_model(
    pretrained=False,
    fine_tune=False, 
    num_classes=len(class_names)
).to(device)
model = model.eval()
print(model)
model.load_state_dict(torch.load('/home/ubuntu/ds-veritone-energy/nhung/Efficientnet/output/cls_logo_efficientnetb1_17112022.pth')['model_state_dict'])

def returnCAM(feature_conv, weight_softmax, class_idx):
    # Generate the class activation maps upsample to 256x256.
    size_upsample = (100, 100)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name, gt_class):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, gt_class, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        pred_class_name = str(class_names[int(class_idx[i])])
        if pred_class_name == gt_class:
            cv2.putText(result, pred_class_name, (20, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(result, pred_class_name, (20, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
        
        # cv2.imshow('CAM', result/255.)
        cv2.waitKey(1)
        cv2.imwrite(f"/home/a4000/Data/nhungnth/car_logo_classify/Efficientnet/output/results/Classify_{save_name}.jpg", orig_image)

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get('features').register_forward_hook(hook_feature)
# Get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

# Define the transforms, resize => tensor => normalize.
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((100, 100)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])
total_test_imgs = 0 
counter = 0
correct_count = 0
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second. 
# Run for all the test images.
total_infer_time = []
test_folder = '/home/ubuntu/ds-veritone-energy/nhung/AICycle-Triton-Engine/output_model/logo'
for logo_folder in os.listdir(test_folder):
    path_imgs = os.path.join(test_folder,logo_folder)
    all_images = glob.glob(path_imgs+'/*.png', recursive=True)

    for image_path in all_images:
        # Read the image.
        total_test_imgs = total_test_imgs+1
        image = cv2.imread(image_path)
        gt_class = logo_folder.upper() #image_path.split(os.path.sep)[-2]
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = orig_image.shape
        # Apply the image transforms.
        image_tensor = transform(image)
        # Add batch dimension.
        image_tensor = image_tensor.unsqueeze(0)
        # Forward pass through model.
        start_time = time.time()
        outputs = model(image_tensor.to(device))
        
        # Get the softmax probabilities.
        probs = F.softmax(outputs).data.squeeze()
        # Get the class indices of top k probabilities.
        class_idx = topk(probs, 1)[1].int()
        print(int(class_idx))
        pred_class_name = str(class_names[int(class_idx)])
        print('Car brand name: '+pred_class_name)
        score = topk(probs, 1).values.item()
        print('score: ',score)
        
        end_time = time.time()
        infer_time = (end_time - start_time)*1000
        print("Inference time: %.2f ms" % infer_time) 
        total_infer_time.append(infer_time)
        # Count collectly predicted samples
        if gt_class == pred_class_name.upper():
            correct_count += 1
        if score < 0.8:
            # Generate class activation mapping for the top1 prediction.
            CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
            # File name to save the resulting CAM image with.
            save_name = f"{image_path.split('/')[-1].split('.')[0]}"+str(pred_class_name)+'_'+str(score)
            # Show and save the results.
            show_cam(CAMs, width, height, orig_image, class_idx, save_name, gt_class)
        counter += 1
        print(f"Image: {counter}")
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
total_infer_time = np.array(total_infer_time)
print(f"Total number of test images: {total_test_imgs}")
print(f"Total correct predictions: {correct_count}")
print(f"Accuracy: {correct_count/total_test_imgs*100:.3f}")
print("Average Inference Time: %.2f miliseconds" % total_infer_time.mean())
# Close all frames and video windows.
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
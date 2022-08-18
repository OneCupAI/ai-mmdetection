import torch
import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


class Mask2Former(object):
    def __init__(self):
        # build the model from a config file and a checkpoint file
        # self.model = init_detector('mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py', 'mask2former.pth', device='cuda:0')
        self.model = init_detector('configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py',
                                   'latest.pth', device='cuda:0')
        self.threshold = 0.5

    def __call__(self, img):
        """
        imgs: (str/ndarray):
           Either image file or loaded image.
        output: (list)
            List of output, each index in list is dict container bbox, score, mask and class

            class_name: the class name
            bbox: ndarray: of shape 1*4 (x1,y1,x2,y2)
            score: float
            mask: ndarray of shape h*w
        """
        dets, labels, masks = inference_detector(self.model, img)

        return dets, labels, masks

def main():
    segModel = Mask2Former()
    testIMG = 'cow.jpg'

    img = cv2.imread(testIMG)

    # cap = cv2.VideoCapture('sit_10.mp4')
    #
    # # Check if camera opened successfully
    # if cap.isOpened() == False:
    #     print("Error opening video stream or file")
    #
    # # Read until video is completed
    # while cap.isOpened():
    #     # Capture frame-by-frame
    #     ret, img = cap.read()
    #     if ret == True:
    #
    #         break
    #
    #     # Break the loop
    #     else:
    #         break
    #
    #
    #
    # img = np.stack([img])
    #
    # img = torch.from_numpy(img)
    #
    # dets, labels, masks = segModel(img)
    img = np.stack([img])

    print(img.shape)

    img = torch.from_numpy(img)

    dets, labels, masks = segModel(img)
    print(masks[0].shape)
    exit(0)



    # ******************** Visualizing Results ********************
    # img = cv2.imread(testIMG, cv2.IMREAD_COLOR)
    cap = cv2.VideoCapture('sit_10.mp4')

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:

            break

        # Break the loop
        else:
            break
    color = (255, 0, 0)
    thickness = 2
    for imageIndex in range(len(dets)):
        det = dets[imageIndex]
        mask = masks[imageIndex]
        label = labels[imageIndex]
        for i in range(0, det.shape[0]):
            topLeft = (int(det[i, 0]), int(det[i, 1]))
            topRight = (int(det[i, 2]), int(det[i, 3]))

            img = cv2.rectangle(img, topLeft, topRight, color, thickness)
            img = cv2.putText(
                img=img,
                text=CLASSES[label[i]],
                org=topLeft,
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=(0, 255, 0)
                , thickness=1)

            expandSeg = torch.unsqueeze(mask[i, ...], 2).cpu().detach().numpy()

            maskedImg = np.where(expandSeg, color, img)

            maskedImg = maskedImg.astype(int)
            img = img.astype(int)
            img = cv2.addWeighted(img, 0.8, maskedImg, 0.2, 0)
        break
    cv2.imwrite('result.jpg', img)


if __name__ == '__main__':
    main()



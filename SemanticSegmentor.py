from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
from tqdm import tqdm

if __name__ == "__main__":
    
    model = 'segformer'
    if model == 'pspnet':
        config_file = 'pspnet_r50-d8_512x1024_40k_cityscapes.py'
        checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    elif model == 'segformer':
        print(' ---------- Semantic Segmentation Model: Model Segformer ---------- ')
        #config_file = 'segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'
        config_file = 'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
        #checkpoint_file = 'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
        checkpoint_file = 'mit_b5_20220624-658746d9.pth'
    else:
        print(' ---------- Please select model! -------- ')
    
    model = init_segmentor(config_file, checkpoint_file, device ='cpu')

    #input_dir = "D:/OpenMMLab/MMSegmentation/images_input"
    #output_dir = "D:/OpenMMLab/MMSegmentation/images_seg"
    input_dir = "./images_input"
    output_dir = "./images_seg"
    input_image_list = []

    for root, dirnames, fnames in os.walk(input_dir):
        for fname in fnames:
            if fname.endswith(('.jpg','.png')):
                full_image_name = os.path.join(root, fname)
                input_image_list.append(full_image_name)
    print('----------- The amount of images: ', len(input_image_list), ' -------------- ')
    for image_file in tqdm(input_image_list):

        image_name = os.path.basename(image_file)
        image = mmcv.imread(image_file)
        result = inference_segmentor(model, image)
        output_image = os.path.join(output_dir, image_name[:-4] + '_seg.png')
        #output_image = image_name
        model.show_result(image, result, out_file = output_image, opacity=1)    
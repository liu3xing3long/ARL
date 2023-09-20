import os
import cv2
from tqdm import tqdm
import concurrent.futures

root_input_path = './images_full'
root_output_path = './images_512'


def resize_image(file_index, filepath_list, IMG_ROW=512, IMG_COL=512):
    img = cv2.imread(fr'{root_input_path}/{filepath_list[file_index]}', cv2.IMREAD_ANYDEPTH)

    print(fr'image shape {img.shape}')
    border_v = 0
    border_h = 0
    if (IMG_COL/IMG_ROW) >= (img.shape[0]/img.shape[1]):
        border_v = int((((IMG_COL/IMG_ROW)*img.shape[1])-img.shape[0])/2)
    else:
        border_h = int((((IMG_ROW/IMG_COL)*img.shape[0])-img.shape[1])/2)
    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    img = cv2.resize(img, (IMG_ROW, IMG_COL))
    cv2.imwrite(fr'{root_output_path}/{filepath_list[file_index]}', img, [cv2.IMWRITE_PNG_COMPRESSION, 0, cv2.IMWRITE_PNG_STRATEGY, 16])


def main():
    filelist = os.listdir(root_input_path)

    with tqdm(total=len(filelist)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(resize_image, f, filepath_list=filelist):f for f in range(len(filelist))}
            for future in concurrent.futures.as_completed(futures):
                filename = filelist[futures[future]]
                try:
                    _ = future.result()
                    pbar.update(1)
                except:
                    print('{} failed.'.format(filename))


if __name__ == '__main__':
    main()
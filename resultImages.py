import shutil
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-t", "--txt", help="the result.txt file", dest="txtFile", default="output/result.txt")
parser.add_argument("-s", "--src", help="src path of image folder", dest="src_path", required=True)
parser.add_argument("-d", "--dst", help="dst path", dest="dst_path", required=True)
args = parser.parse_args()


count = 0
# with open('output/result.txt', 'r') as f:
# with open('data/old/image_test_new_label.txt', 'r') as f:
with open(f'{args.txtFile}', 'r') as f:
    for row in f:
        row = row.strip()
        imageName, groupID = row.split(' ')
        # src_path = './data/image_train_augmented/' + imageName
        # dst_path = './result images/' + groupID
        # src_path = 'data/old/image_test_new/' + imageName
        # dst_path = 'data/old/result images/' + groupID
        src_path = args.src_path + '/' + imageName
        dst_path = args.dst_path + '/' + groupID

        # 若destination資料夾不存在則建立資料夾
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        shutil.copy(src_path, dst_path)

        count += 1
print(f'Finish processing {count} images.')
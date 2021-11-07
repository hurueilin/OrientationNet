import shutil
import os

count = 0
with open('output/result.txt', 'r') as f:
    for row in f:
        row = row.strip()
        imageName, groupID = row.split(' ')
        src_path = './data/image_train_augmented/' + imageName
        dst_path = './result images/' + groupID
        
        # 若資料夾不存在則建立資料夾
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        shutil.copy(src_path, dst_path)

        count += 1

print(f'Finish processing {count} images.')
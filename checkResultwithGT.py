import shutil

def isNeighbor(gt, pred):
    if abs(gt - pred) == 1 or (gt==9 and pred==0) or (gt==0 and pred==9):
        return True
    else:
        return False

with open ('data/syn_test_label.txt', 'r') as f1:
    with open('output/result.txt', 'r') as f2:
        count = 0
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            imageName, gt = line1.split()

            line2 = line2.strip()
            imageName, pred = line2.split()

            gt, pred = int(gt), int(pred)
            if gt != pred:
                # print(f'{imageName} is not match => gt={gt}, predict={pred}')
                # shutil.copy('data/image_train_augmented/'+imageName, 'output/wrongPreds')
                # count += 1
                if not isNeighbor(gt, pred):
                    print(f'{imageName} is not match => gt={gt}, predict={pred}')
                    shutil.copy('data/image_train_augmented/'+imageName, 'output/Hard_wrongPreds')
                    count += 1
print('Wrong predictions count:', count)

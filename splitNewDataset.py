# Read all label data in orientation_label_augmented.txt
groups = [[] for i in range(10)]  # save imageNames of each groupID
count = 0
with open('data/orientation_label_augmented.txt', 'r') as f:
    for row in f:
        row = row.strip()
        imageName, groupID = row.split()

        groups[int(groupID)].append(imageName)
        count += 1
print(f'{count} items are saved in list.')


print('Creating new train/val/test labels...')
with open('data/syn_train_label.txt', 'w') as fTrain:
    with open('data/syn_val_label.txt', 'w') as fVal:
        with open('data/syn_test_label.txt', 'w') as fTest:
            for groupID in range(10):
                # Train (the first 5000 pic)
                for imageName in groups[groupID][:5000]:
                    print(f'{imageName} {groupID}', file=fTrain)
                # Val
                for imageName in groups[groupID][5000:5500]:
                    print(f'{imageName} {groupID}', file=fVal)
                # Test
                for imageName in groups[groupID][5500:5800]:
                    print(f'{imageName} {groupID}', file=fTest)
print('Finished')
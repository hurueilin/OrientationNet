import os

files = os.listdir('data/image_test_new')
print(len(files))

with open('data/name_test_new.txt', 'w') as f:
    count = 0
    for filename in files:
        print(filename, file=f)
        count += 1
print(f'Finish creating {count} names.')

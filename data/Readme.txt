<old>
training data: 合成資料image_train_augmented(206,968)
testing data: 先拿真實資料image_test_new(16,675)試試看

註：
* image_train_augmented: 包含將14818張group0水平翻轉成group4後的圖片，共206968張
* orientation_label_augmented.txt為其label檔
合成資料192150 + 14818 = 206968 images after augmented

===================================================================================
<11/06>
把合成資料image_train_augmented(206,968)拆成train/val/test
* syn_train_label: 5000*10=50000張
* syn_val_label:  500*10=5000張
* syn_test_label: 300*10=3000張

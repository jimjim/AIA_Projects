## 埋葬蟲移動軌跡偵測

### beetle-tracking Datasets

1. Detection_data: (**26** videos + **26** txt files)
    txt file meaning:
    ['frame_no, [(type, (Xmin, Ymin),(Xmax, Ymax), (type, (Xmin, Ymin),(Xmax, Ymax))]',
    'frame_no, [(type, (Xmin, Ymin),(Xmax, Ymax)]',
    ...]
    frame_no: the frame number, start from 1
    type: the type of tags on beetle, [1, 2, 3, 4, 5] for [O, X, =, A, unknow]
    (Xmin, Ymin),(Xmax, Ymax):bounding box of the beetle

2. classification_data:
    beetles' photos with 4 classes tags [0, 1, 2, 3] for [O, X, =, A]
    train: from other videos
    test: from detection data

### utilities
1. utility/extractor.ipynb
轉換原有資料集成VOC格式
2. utility/extractor_w_classification.ipynb
驗證resnet模型在detection資料集上的分類結果
3. utility/video_crop.ipynb
切割video工具 (by frame index)
4. utility/sp.py
切割video指令（by time)



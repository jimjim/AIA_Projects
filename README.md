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




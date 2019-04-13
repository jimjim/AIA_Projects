# 埋葬蟲移動軌跡偵測  Project_JT072_09_Biodiv-Beetle
於影片(1280x720)中**識別**及**追踪**背上不同圖形的埋葬蟲(共4隻)

![output_YoloTrackingClassify](images/output_YoloTrackingClassify.gif)

## Code structure

#### DeepLabCut/
1. `ByPackage/BeetleTracking-Lindo-2019-03-26/` 含在可執行 wx 環境 label 好的資料及專案設定 `config.yaml`
2. `ByPackage/dlc-beetleTracking.ipynb` 以 python package 方式使用 DeepLabCut Toolbox 訓練追踪埋葬蟲

#### classfication/
1. `transfer_resnet50-0321-2.ipynb` 運用 ResNet50 建立 `/datasets/beetle-tracking/classification_data` 的訓練模型

#### keras-yolo3-master/
 - 修改 Yolo3 適用於埋葬蟲資料並加入tracking 與 classify，詳細執**行步驟與使用說明**請參閱 [keras-yolo3/README](/keras-yolo3-master/README.md)

#### utility/
 1. `extractor.ipynb` 轉換原有資料集成 VOC 格式
 2. `extractor_w_classification.ipynb` 驗證 ResNet 模型在 detection 資料集上的分類結果
 3. `video_crop.ipynb` 切割 video 工具(by frame index, ROI to crop)
 4. `sp.py` 分割 video 指令(by time)

## Next ...
- [ ] DeepLabCut 多動物行為分析
- [ ] 修正 classification 模型

## Reference

- [afunTW / beetle-tracking](https://github.com/afunTW/beetle-tracking
)
- [DeepLabCut](https://github.com/AlexEMG/DeepLabCut)
- [End-to-end people detection in crowded scenes](https://arxiv.org/abs/1506.04878)
- [experiencor / keras-yolo3](https://github.com/experiencor/keras-yolo3)
- [Nvidia - AI Enables Markerless Animal Tracking](https://news.developer.nvidia.com/ai-enables-markerless-animal-tracking/)
- [Object tracking](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv)
- [Simple online and realtime tracking (SORT)](https://github.com/abewley/sort)
- [Tracking multiple high-density homogeneous targets](http://www.eecs.qmul.ac.uk/~andrea/thdt.html)

 


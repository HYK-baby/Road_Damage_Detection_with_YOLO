# Road_Damage_Detection_with_YOLO

rddDetector has only 3.35M Params.
![structure](https://github.com/HYK-baby/Road_Damage_Detection_with_YOLO/blob/main/structure.png)

Trained on Global Road Damage Detection 2020.
Model|backbone|params(M)|inputs|GFLOPs|test1|test2
---- | ------ | ------- | ---- | ---- | --- | ---

YOLOv5-X|Modified CSP v5|86.7|640×640|205.7|58.14%|57.51%
EfficientDet-D3|Efficient-B3|12.0|896×896|25.0|56.5%|54.7%
Faster R-CNN w.FPN|Resnet 101|60.0|1333×1333|246.0|53.68%|54.26%
YOLOv4-Tiny|Modified CSP-Tiny|5.87|608×608|16.04|42.85%|41.16%
YOLOX-Tiny|Modified CSP v5-Tiny|5.06|608×608|15.06|46.56%|45.69%
Ours|MobileNetv3|4.72|640×640|7.45|46.93%|47.24%
Ours|SC-DenseNet|3.35|640×640|7.33|54.04%|52.23%
Ours w.Mosiac,Mixup|SC-DenseNet|3.35|640×640|7.33|55.58%|55.00%

# Yolov8-ncnn
convert yolov8 to ncnn
##### Download model
you can download yolov8 model from https://github.com/ultralytics/ultralytics/tree/15b3b0365ab2f12993a58985f3cb7f2137409a0c
##### Modify 'forward' methods in 'ultralytics/ultralytics/nn/modules.py'
- For detection
1. class C2f(nn.Module)
 ```python
     def forward(self, x):
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))
        # !< https://github.com/FeiGeChuanShu/ncnn-android-yolov8
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))
 ```
 2. ```class Detect(nn.Module)```


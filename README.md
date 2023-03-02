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
```python
    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        # box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # y = torch.cat((dbox, cls.sigmoid()), 1)
        # return y if self.export else (y, x)
        # !< https://github.com/FeiGeChuanShu/ncnn-android-yolov8
        pred = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).permute(0, 2, 1)
        return pred
```
- For Segmentation
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
```python
    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        # box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # y = torch.cat((dbox, cls.sigmoid()), 1)
        # return y if self.export else (y, x)
        # !< https://github.com/FeiGeChuanShu/ncnn-android-yolov8
        pred = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # different with detection
        return pred
```
3. ```class Segment(Detect)```
```python
    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        # return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
        # !< https://github.com/FeiGeChuanShu/ncnn-android-yolov8
        return (torch.cat([x, mc], 1).permute(0, 2, 1), p.view(bs, self.nm, -1)) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
```

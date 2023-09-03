### 肺实质分割

##### 训练数据集：

来源于[**肺纤维化进展**的数据集](：https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/overview)（预测肺功能下降）

与之对应的与肺部功能相关的[重要器官的分割数据集](https://www.kaggle.com/datasets/sandorkonya/ct-lung-heart-trachea-segmentation) 后者由前者数据标注而成，而且标注数量 **<** 原始数据

##### 模型训练：

将数据集分为4: 1，4份用于训练，1份用于验证

```shell
python train_baseline.py
```

经过200轮次的训练后，得到`demo.pth`模型

   

##### 模型验证：

```
python model_test_std.py
```

验证的Dice = 0.98

##### 模型推理：

```
python model_infer.py
```

以呼气、吸气CT为测试图像，推理结果在`test_pred`目录下


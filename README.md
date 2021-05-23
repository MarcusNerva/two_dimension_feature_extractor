#二维特征提取

实现的模型放在了`./models`下，如果想要添加新的模型，请也添加在该包下。
该项目的feature extractor是基于torchvision 现成的模型实现的。

在添加完毕后，要在`./models/__init__.py`中的"模型工厂"`model_factory`函数中进行注册。
运行`main.py`即可完成特征提取，不过要填写好相应的参数。提取出来的特征放在`./results`文件夹下。
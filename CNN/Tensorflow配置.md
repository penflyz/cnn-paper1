# Tensorflow配置

1. 创建环境

   ```
   conda create -n new36tf python=3.6
   conda list
   ```

备注：**更换源可能无法创建环境**，需要恢复默认源 

   ```
   conda config --remove-key channels
   ```

   ```
   pip install wrapt six grpcio protobuf numpy gast keras-preprocessing termcolor werkzeug markdown wheel absl-py tensorboard google-pasta astor tensorflow-estimator h5py keras-applications matplotlib sklearn tensorflow opencv-python
   ```

   **利用pip安装的话，记得加版本号。**

* 更换源

  * 用户\AppData\Roaming\pip\

    ```
    [global]
    index-url=http://pypi.douban.com/simple
    extra-index-url=
    	http://mirrors.aliyun.com/pypi/simple/
    	https://pypi.tuna.tsinghua.edu.cn/simple/
    	http://pypi.mirrors.ustc.edu.cn/simple/
    
    [install]
    trusted-host=
    	pypi.douban.com
    	mirrors.aliyun.com
    	pypi.tuna.tsinghua.edu.cn
    	pypi.mirrors.ustc.edu.cn
    ```

  保存为“pip.ini”

# TensorFlow2.0降到1.0

```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```


# Interpretable Convolutional Neural Networks


# Introduction

This paper proposes a method to modify traditional convolutional neural networks (CNNs) into interpretable CNNs, in order to clarify knowledge representations in high conv-layers of CNNs. In an interpretable CNN, each filter in a high conv-layer represents a certain object part. We do not need any annotations of object parts or textures to supervise the learning process. Instead, the interpretable CNN automatically assigns each filter in a high conv-layer with an object part during the learning process. Our method can be applied to different types of CNNs with different structures. The clear knowledge representation in an interpretable CNN can help people understand the logics inside a CNN, i.e., based on which patterns the CNN makes the decision. Experiments showed that filters in an interpretable CNN were more semantically meaningful than those in traditional CNNs.

# Citation

Please cite the following paper, if you use this code.
Quanshi Zhang, Ying Nian Wu, and Song-Chun Zhu, "Interpretable Convolutional Neural Networks," in CVPR 2018

# Code

We released the code with slight technical extensions to the above paper for more robustness. For example, the code learned the parameter \beta instead of simply setting \beta=4.

We will release the code based on PyTorch and TensorFlow, later.

# How to use

run demo.m

Note that please set in the window of the MATLAB following "HOME --> Preferences --> MATLAB --> General --> MAT-Files --> MATLAB Version 7.3 or later." Thus, the Matlab can save large MAT files.

Please see demo.m for detailed introduction of the code

# 资源开销

| `run demo.m` | train阶段 | val阶段 |
| ------------ | --------- | ------- |
| 显存/MB      | 10,845    | 306     |
| 时间/min     |           |         |
|              |           |         |



# 老梁的日志

## 报错

### Attempt to execute SCRIPT vl_nnconv as a function

执行

```bash
cd ./code
matlab
```

打开matlab后

```matlab
run demo
```

报错

```
Attempt to execute SCRIPT vl_nnconv as a function
```

原因是 [参考](http://www.vlfeat.org/matconvnet/faq/) matlab 的神经网络工具包（MatConvNet）没有编译。

工具包在`./matconvnet-1.0-beta24/`，编译方法如下

* 在matlab中执行如下编译命令

* ```matlab
  cd ./matconvnet-1.0-beta24/matlab/
  vl_compilenn('enableGpu', true, ...
                 'cudaRoot', '/Developer/NVIDIA/CUDA-8.0', ...
                 'cudaMethod', 'nvcc')
  ```

* 之后每次启动`matlab`，请加前缀

  ```shell
  LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6" matlab
  ```

  不加的话，会报错

  ```
  /usr/local/MATLAB/R2014a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6:version 'GLIBCXX_3.4.21' not found
  ```

* 之后每次使用MatConvNet前，需将 MatConvNet 加到 MATLAB 搜索路径

  ```matlab
  run ./matconvnet-1.0-beta24/matlab/vl_setupnn
  ```

  注：执行 `run demo` 前无需手动执行上句，因为在代码中做了

* 测试

  ```matlab
  vl_testnn('gpu', true)
  ```

  如不报错（例如下），则为编译成功

  ```
  Totals:
     3366 Passed, 0 Failed, 0 Incomplete.
     1802.9089 seconds testing time.
  ```

### imdb 没有 images 属性

执行

```bash
cd ./code
matlab
```

打开matlab后

```matlab
run demo
```

报错

```
Error in learn_icnn (line 58)
disp(imdb.images)

Error in demo (line 48)
    learn_icnn(model,categoryName,dropoutRate);
```

一次解决

```shell
cd code/mat
rm n02118333/imdb.mat
```

永久解决方法：matlab中执行

```
save(opts.imdbPath,'-struct','imdb');
```

改为

```
save(opts.imdbPath,'-struct','imdb','-v7.3');
```

### Out of memory on device.

解决办法：matlab下

```xmatlab
gpuDevice(<要用的显卡编号>)
```

此时`gpustat`看见自己用了指定的那张显卡，然后再执行原先的matlab命令

注意：matlab中"<要用的显卡编号>"从1开始，与`nvidia-smi`从0开始不同

### You are using gcc version '5.4.0'. 

运行`run demo`，报错

```
Warning: You are using gcc version '5.4.0'. The version of gcc is not supported. The version currently supported with MEX is '4.7.x'.
```

### 不显示warning

matlab中执行

```
warning off
```

# 运行方法

## 编译

首次使用，需要编译

### 编译`/code/tool/edges-master/piotr_toolbox/`

```shell
cd ./code
matlab
```

matlab中执行

```matlab
toolboxCompile
```

会报错

```
Warning: You are using gcc version '5.4.0'. The version of gcc is not supported. The version currently supported with MEX is '4.7.x'.
```

但不影响编译

### 编译`./matconvnet-1.0-beta24/`

工具包在`./matconvnet-1.0-beta24/`，编译方法如下

- 在matlab中执行如下编译命令

- ```matlab
  cd ./matconvnet-1.0-beta24/matlab/
  vl_compilenn('enableGpu', true, ...
                 'cudaRoot', '/Developer/NVIDIA/CUDA-8.0', ...
                 'cudaMethod', 'nvcc')
  ```

- 之后每次启动`matlab`，请加前缀

  ```shell
  LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6" matlab
  ```

  不加的话，会报错

  ```
  /usr/local/MATLAB/R2014a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6:version 'GLIBCXX_3.4.21' not found
  ```

- 之后每次使用MatConvNet前，需将 MatConvNet 加到 MATLAB 搜索路径

  ```matlab
  run ./matconvnet-1.0-beta24/matlab/vl_setupnn
  ```

  注：执行 `run demo` 前无需手动执行上句，因为在代码中做了

- 测试

  ```matlab
  vl_testnn('gpu', true)
  ```

  如不报错（例如下），则为编译成功

  ```
  Totals:
     3366 Passed, 0 Failed, 0 Incomplete.
     1802.9089 seconds testing time.
  ```

## 运行

之后使用不用编译

```shell
cd ./code
matlab
```

matlab中运行

```
warning off
run demo
```

## 修改参数


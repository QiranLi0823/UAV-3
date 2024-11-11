# 文档说明

## 1. 环境配置
按照requirements.txt和require.txt(为防止安装依赖包出现不兼容问题，参考require.txt时从后往前安装，缺什么依赖包在两个txt中都能找到)


## 2. 文件说明

- 2.1 文件树如下
  -----workpath
    |-apex                  # 相关依赖包
    |-data                  # 用于生成angle必要数据
    |-infogcn(FR_Head)      # 存放infogcn(Head)模型完整代码
    |-mixformer             # 存放mixformer模型完整代码
    |-STTFormer(UAV)        # 存放STTFormer(UAV)模型完整代码
    |-GS-A and GS-B         # 存放三个模型训练结果，A为官方提供的val_joint生成的pkl文件，B为test_joint生成的pkl文件
    |-torchligth            # 相关依赖包
    |-ensemble.py           # 融合模型训练的权重
    |-requirement.txt       # 环境所需要的依赖库
    |-require.txt           # 环境所需要的依赖库
    |-README.md             # 对本项目代码的说明
  

## 3. 数据位置
- 3.1 '文件说明'部分已说明数据集存放的位置
- 3.2 参数设置均在config文件中

## 4. 运行方式

- 4.1 在运行代码前，请先配置好'环境配置'中所需要的依赖的安装包
- 4.2 安装apex，获取地址(https://github.com/NVIDIA/apex.git)
- 4.3 相关数据模块的训练指令根据config文件中的运行指令
- 4.4 运行时要确定相关数据路径是否正确

注:安装apex时，在utils.py中将（if cached_x.grad_fn.next_functions[1][0].variable is not x:）修改为（if cached_x.grad_fn.next_functions[0][0].variable is not x:），然后在用指令pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   
   相关数据训练时注意在config文件中修改路径，训练测试的指令在config文件中

## 5. 结果复现流程

- 5.1 确定环境配置以及相关依赖包以成功安装
- 5.2 将官方提供的数据集data放到与UAV-2同级目录下。
- 5.3 数据预处理，处理官方所给的train_joint中所有的空数据，相关空数据处理的代码在data_process.ipynb中
- 5.4 利用UAV-2文件中的data目录下的gen_angle_data.py运行文件生成angle模态（注意在代码中修改数据集路径）
- 5.5 将infogcn(FR_Head)中的所有模态分别训练，得到的训练日志，训练权重均保存在work_dir下，验证结果在temp目录下
- 5.6 将mixformer中的(_1,_2,_6,angle_1,motion_1,motion_2,motion_6)分别训练得到的训练日志，训练权重，测试结果均保存在temp文件夹下
- 5.7 将STTFormer中的(angle,bone,joint,motion)分别训练得到的训练日志，训练权重均保存在temp文件夹下,测试的结果在new_work_dir下
- 5.8 对结果运行ensemble.py操作融合，融合后生成B数据集的置信度文件

```.
注:每个模型的config中都标有运行指令,运行时修改训练或者测试的路径
   对于infogcn中angle模态的生成，要注意生成的数据shape,代码中有相关注释，在训练infogcn时，需要在feeder.uav.py中更改空数据填充的shape与之对应
   32frame_1测试时data_numpy = np.zeros((3,32,17,2))    128frame_1测试时data_numpy = np.zeros((3,128,17,2))   angle_1测试时data_numpy = np.zeros((9,64,17,2))
   对于mixformer,STTFormer生成的train_angle.shape(16723,9,17,300,2),test_angle    val_angle.shape(2000,9,17,300,2)   test_angle.shape(4307,9,17,300,2)
   对于infogcn中的train_angle_(FR_Head).shape(16723,9,300,17,2),val_angle_(FR_Head).shape(2000,9,300,17,2)    test_angle(4307,9,300,17,2)
   相关生成angle.shape在gen_angle_data.py中也有注释
   我们将所有的训练结果整合放在了GS-A和GS-B文件夹下，便于最终融合时读取(GS-A是自己验证模型的,GS-B是用于官方测试)
   对于B数据集的测试标签为自己做的标签，运行时自己生成一个B测试集的标签以便代码正常运行，生成标签的代码在data_process.ipynb中
   融合时注意数据路径
```


## 6. 源代码位置
- 6.1 (https://github.com/QiranLi0823/UAV-2)
- 6.2 我们自己训练的权重在github链接上均可下载



注:模型参考和运行指令参考链接(https://github.com/happylinze/UAV-SAR)


### training
infogcn(FR_Head)

1.we added the FR_Head module into infoGCN when training the model. And we trained the k=1, k=2, and k=6. You could use the commands as:

cd ./infogcn(FR_Head)

# k=1 use FR_Head
python main.py --config ./config/uav_csv2/FR_Head_1.yaml --work-dir <the save path of results> 

# k=2 use FR_Head
python main.py --config ./config/uav_csv2/FR_Head_2.yaml --work-dir <the save path of results>
    
# k=6 use FR_Head
python main.py --config ./config/uav_csv2/FR_Head_6.yaml --work-dir <the save path of results>


2.We also trained the motion by k=1, k=2 and k=6. You could use the commands as:

cd ./infogcn(FR_Head)

# By motion k=1
python main.py --config ./config/uav_csv2/motion_1.yaml --work-dir <the save path of results>

# By motion k=2
python main.py --config ./config/uav_csv2/motion_2.yaml --work-dir <the save path of results>

# By motion k=6
python main.py --config ./config/uav_csv2/motion_6.yaml --work-dir <the save path of results>


3.The default sample frames for model is 64, we also trained the 32 frames and the 128 frames. The commands as:

cd ./infogcn(FR_Head)

# use 32 frames
python main.py --config ./config/uav_csv2/32frame_1.yaml --work-dir <the save path of results>

# use 128 frames
python main.py --config ./config/uav_csv2/128frame_1.yaml --work-dir <the save path of results>


4.After get the MMVRAC_CSv1_angle.npz and MMVRAC_CSv2_angle.npz, we trained the data by the command as:

cd ./infogcn(FR_Head)

# use angle to train
python main.py --config ./config/uav_csv2/angle_FR_Head_1.yaml --work-dir <the save path of results>

cd ./infogcn(FR_Head)


5.We also tried the FocalLoss to optimize the model. The command as:

# use focalloss
python main.py --config ./config/uav_csv2/focalloss_1.yaml --work-dir <the save path of results>

mixformer

1. We trained the model in k=1, k=2 and k=6. You could use the commands as:
```shell
cd ./mixformer

# k=1
python main.py --config ./config/uav_csv2/_1.yaml --work-dir <the save path of results>

# k=2
python main.py --config ./config/uav_csv2/_2.yaml --work-dir <the save path of results>

# k=6
python main.py --config ./config/uav_csv2/_6.yaml --work-dir <the save path of results>
```

2. We also trained the model in k=1, k=2 and k=6 with motion. The commands as:
```shell
cd ./mixformer

# By motion k=1
python main.py --config ./config/uav_csv2/motion_1.yaml --work-dir <the save path of results>

# By motion k=2
python main.py --config ./config/uav_csv2/motion_2.yaml --work-dir <the save path of results>

# By motion k=6
python main.py --config ./config/uav_csv2/motion_6.yaml --work-dir <the save path of results>
```

3. And we tried the angle data to train the model. The command as:
```shell
cd ./mixformer

# use angle
python main.py --config ./config/uav_csv2/angle_1.yaml --work-dir <the save path of results>

```


STTFormer

We trained joint, bone and motion. The commands as follows:
```shell
cd ./sttformer

# train joint
python main.py --config ./config/uav_csv2/joint.yaml --work-dir <the save path of results>

# train bone
python main.py --config ./config/uav_csv2/bone.yaml --work-dir <the save path of results>

# train motion
python main.py --config ./config/uav_csv2/motion.yaml --work-dir <the save path of results>

# train angle
python main.py --config ./config/uav_csv2/angle.yaml --work-dir <the save path of results>

```

### Testing
If you want to test any trained model saved in `<work_dir>`, run the following command: 
```shell
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save_score True --weights <work_dir>/xxx.pt
```

最终结果整合保存为形式为
```.

- GS-A/
   - infogcn/
   - 32frame_1/
      - epoch1_test_score.pkl
   - 128frame_1/
      - epoch1_test_score.pkl
   - angle_1/
      - epoch1_test_score.pkl
   -FocalLoss_1/
      - epoch1_test_score.pkl
   - FR_Head_1/
      - epoch1_test_score.pkl
   - FR_Head_2/
      - epoch1_test_score.pkl
   - FR_Head_6/
      - epoch1_test_score.pkl
   - motion_1/
      - epoch1_test_score.pkl
   - motion_2/
      - epoch1_test_score.pkl
   - motion_6/
      - epoch1_test_score.pkl  
   - mix_1/
      - epoch1_test_score.pkl
   - mix_2/
      - epoch1_test_score.pkl
   - mix_6/
      - epoch1_test_score.pkl
   - mix_angle_1/
      - epoch1_test_score.pkl
   - mix_motion_1/
      - epoch1_test_score.pkl
   - mix_motion_2/
      - epoch1_test_score.pkl
   - mix_motion_6/
      - epoch1_test_score.pkl
   - ntu60_A
      - epoch1_test_score.pkl
   - ntu60_B
      - epoch1_test_score.pkl
   - ntu60_J
      - epoch1_test_score.pkl
   - ntu60_M
      - epoch1_test_score.pkl   
         ...
- GS-B/
   - infogcn/
   - 32frame_1/
      - epoch1_test_score.pkl
   - 128frame_1/
      - epoch1_test_score.pkl
   - angle_1/
      - epoch1_test_score.pkl
   -FocalLoss_1/
      - epoch1_test_score.pkl
   - FR_Head_1/
      - epoch1_test_score.pkl
   - FR_Head_2/
      - epoch1_test_score.pkl
   - FR_Head_6/
      - epoch1_test_score.pkl
   - motion_1/
      - epoch1_test_score.pkl
   - motion_2/
      - epoch1_test_score.pkl
   - motion_6/
      - epoch1_test_score.pkl  
   - mix_1/
      - epoch1_test_score.pkl
   - mix_2/
      - epoch1_test_score.pkl
   - mix_6/
      - epoch1_test_score.pkl
   - mix_angle_1/
      - epoch1_test_score.pkl
   - mix_motion_1/
      - epoch1_test_score.pkl
   - mix_motion_2/
      - epoch1_test_score.pkl
   - mix_motion_6/
      - epoch1_test_score.pkl
   - ntu60_A
      - epoch1_test_score.pkl
   - ntu60_B
      - epoch1_test_score.pkl
   - ntu60_J
      - epoch1_test_score.pkl
   - ntu60_M
      - epoch1_test_score.pkl
      ...
```
Then run the command as:
```shell
python ensemble.py
```

# Citation
```.
@inproceedings{li2024hybrid,
  title={A Hybrid Multi-Perspective Complementary Model for Human Skeleton-Based Action Recognition},
  author={Li, Linze and Zhou, Youwei and Hu, Jiannan and Wu, Cong and Xu, Tianyang and Wu, Xiao-Jun},
  booktitle={2024 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

# Python 环境

## 1. 官网集合

> + Python官网：<https://www.python.org>
> + Anaconda官网：<https://www.anaconda.com>
> + Pip官网：<https://pypi.org>
> + Poetry官网：<https://python-poetry.org>
> + Python项目管理：<https://packaging.python.org/en/latest/tutorials/packaging-projects/>

> + 官方文档：<https://docs.python.org/zh-cn/3.9/howto/>

## 2. `Python`安装

> 这里以3.10.11版本为例
> + Win安装版本：<https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe>
> + Win内嵌版本：<https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip>
> + Linux版本：<https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tar.xz>

> **注意事项**
> + 在机器学习过程中会发现许多模块缺失，建议直接使用安装版本的或使用`Anaconda`集成环境。
> + 本文中所有的`APP_HOME`都是指代项目或应用的根目录。
> + 如果环境变量，本文中的所有`python -m`均可以去掉。

### 2.1. Windows安装

#### 2.1.1. 环境配置

```shell
# Windows PowerShell
$Env:PYTHONHOME="D:\Software\Python310"
$Env:PYTHONPATH="$Env:PYTHONHOME\Lib\site-packages"
$Env:Path="$Env:PYTHONHOME;$Env:PYTHONHOME\Scripts;$Env:Path;"
```

#### 2.1.2. 执行文件安装

```shell
# 下载后，安装至`$Env:PYTHONHOME`所指路径即可
curl -OL "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
```

#### 2.1.3. 内嵌版本安装

```shell
# 创建并进入`Python`的安装路径
mkdir "$Env:PYTHONHOME"; cd "$Env:PYTHONHOME"

# 下载相关文件
curl -OL "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
curl -OL "https://bootstrap.pypa.io/get-pip.py"

# 解压文件(这里是java环境带的jar命令, 可自行用其他软件解压)
jar xvf .\python-3.10.11-embed-amd64.zip
# 安装`pip`
python get-pip.py

# 在`python3xx._pth`文件最后加上`Lib/site-packages`
# 这里`python-3.10.11`对应的文件名是`python310._pth`
echo "Lib/site-packages" >> python310._pth
```

### 2.2. Linux安装

#### 2.2.1. 环境配置

```shell
PYTHONHOME="/home/tools/python310"
LD_LIBRARY_PATH="$PYTHONHOME/lib:$LD_LIBRARY_PATH"
PATH="$PYTHONHOME/bin:$PATH"
export PYTHONHOME LD_LIBRARY_PATH PATH
```

> + 在项目中，可以通过`PYTHONPATH`环境变量指定加载其他路径的`model`
> + 例如读取应用路径`APP_HOME`下`lib/site-packages`中的`model`

```shell
export PYTHONPATH="$APP_HOME/lib/site-packages:$PYTHONHOME"
```

#### 2.2.2. 下载安装

> **注意事项**：下面安装需要有外网环境, 如内网环境安装有两种方案
> 1. 提前准备好所有需要外网下载的资源，然后按需编译安装；
> 2. 准备一个与内网环境相近(最好一致)的虚拟环境，按下面过程编译后，直接打包移植；

```shell
# 如果没有外网, 需要自行编译安装或在`yum源`找齐所需包和依赖安装
yum install -y libffi-devel
#curl -OL https://github.com/libffi/libffi/releases/download/v3.4.6/libffi-3.4.6.tar.gz
#LD_LIBRARY_PATH="/home/tools/libffi-3.4.4/lib64:$LD_LIBRARY_PATH"
#./configure --prefix=/home/tools/libffi-3.4.4 --disable-static --with-gcc-arch=native

curl -OL 'https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz'
curl -OL 'https://bootstrap.pypa.io/get-pip.py'

# 执行`./configure`会提示使用`--enable-optimizations`
# 但部分系统不支持该参数(`make`会出错), 建议去掉
./configure --prefix=$PYTHONHOME --with-ssl-default-suites=openssl --enable-shared
make && make install

# 常用模块安装(无外网环境可忽略, 在项目发布时一起打包好所需依赖就行)
python3 -m pip install -U pip setuptools wheel poetry -i "https://mirrors.aliyun.com/pypi/simple"
```

### 3. `Pip`包管理

```shell
# 列出所有安装的包
python -m pip list

# 安装指定的`model`, 以及`model`的依赖
python -m pip install <model...>

# 安装并更新指定的`model`, `--upgrade`或`-U`表示更新
python -m pip install --upgrade <model...>

# 只安装`model`本身, 不安装其他依赖
python -m pip install <model...> --no-deps

# 指定安装`model`的源(镜像), 加速安装
python -m pip install <model...> -i <source>

# 指定安装`model`的路径(`file://`,`https://`)
python -m pip install <model...> -f <url>

# 移除指定的`model`(不会移除依赖的其他`model`)
python -m pip uninstall <model...>
```

```shell
# 将`module`关系导出到`requirements.txt`
python -m pip freeze > requirements.txt

# 安装`requirements.txt`指定`module`
# 建议在虚拟环境执行，避免环境污染
python -m pip install -r requirements.txt
```

> 更多用法参考<https://pip.pypa.io/en/latest/user_guide/>
> + 安装模块：<https://pip.pypa.io/en/latest/user_guide/#installing-packages>
> + 卸载模块：<https://pip.pypa.io/en/latest/user_guide/#uninstalling-packages>
> + 列出模块：<https://pip.pypa.io/en/latest/user_guide/#listing-packages>
> + 查询模块：<https://pip.pypa.io/en/latest/user_guide/#searching-for-packages>
> + `.whl`安装：<https://pip.pypa.io/en/latest/user_guide/#installing-from-wheels>
> + `requirements.txt`：<https://pip.pypa.io/en/latest/user_guide/#requirements-files>

## 4. `virtualenv`虚拟环境

> + 不同项目有不同的依赖环境和版本要求，virtualenv`或`venv`用于项目环境隔离；
> + 非内嵌版本尽管以包含`venv`模块, 仍然建议安装`virtualenv`模块；

```shell
# 安装虚拟环境
python -m pip install virtualenv
```

```shell
# `APP_HOME`指代项目或应用的根目录
cd $APP_HOME

# 初始虚拟环境, `.venv`为虚拟环境存放路径
python -m virtualenv .venv

# Windows 激活虚拟环境
.venv\Scripts\activate

# Linux 激活虚拟环境
source .venv/bin/activate
```

## 5. `Poetry`项目管理

### 5.1. `Poetry`安装

```shell
# 安装`poetry`
python -m pip install poetry

# 查看`poetry`配置
python -m poetry config --list

# 设置虚拟环境到项目里(非常建议, 避免项目环境交叉)
python -m poetry config virtualenvs.in-project true

# 设置缓存路径(系统盘空间不够可以调整该项)
python -m poetry config cache-dir "D:\\Temp\\Pypoetry\\Cache"
```

### 5.2. 初始化项目

```shell
# 创建新项目
python -m poetry new poetry-demo

# 初始已有项目
python -m poetry init
```

### 5.3. 项目管理

```shell
# 安装`pyproject.toml`中的依赖
python -m poetry install

# 升级`pyproject.toml`中的依赖
python -m poetry update

# 显示`pyproject.toml`中的依赖
python -m poetry show

# 添加`model`
python -m poetry add <model...>

# 从指定源添加`model`
python -m poetry add <model...> --source=<source>
```

### 5.4. 其他用法

```shell
# 添加源
poetry source add tsinghua https://pypi.tuna.tsinghua.edu.cn/simple
poetry source add aliyun https://mirrors.aliyun.com/pypi/simple/

# 开启`virtualenv`环境
python -m poetry shell

# 在`virtualenv`环境运行脚本
python -m poetry run python src/demo.py
```

> + 官方文档-poetry: <https://python-poetry.org/docs/basic-usage/>
> + 参考文档-poetry: <https://notes.zhengxinonly.com/environment/use-poetry.html>

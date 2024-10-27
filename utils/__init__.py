# 显式识别为包
# 代码会自动执行 初始化一些操作
# 可以通过__all__控制 from package import * 行为
# 可以直接导入包内的模块

from .misc import *

# 可以直接通过 myproject 导入 misc.py 中的所有内容：
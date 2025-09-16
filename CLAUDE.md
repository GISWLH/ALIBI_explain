这是一个ALIBI结合机器学习的可解释文档

运行代码所需的环境在conda activate geo，因此运行py代码前先激活环境
timeout xx其中xx应该大于10分钟，因为代码运行较慢，可以多给一些时间
我不喜欢定义太过复杂的函数，并运行main函数，我是深度jupyter notebook用户，我喜欢直接的代码，简单的函数定义是可以接受的
使用matplotlib可视化，绘图使用Arial字体(在linux中手动增加我们的arial字体），绘图中的图片标记都用英文，我更喜欢seaborn高级封装格式
代码都放到code文件夹下
结果图都放到figure
数据是geo.csv，实战时用这个数据，径流预测模拟

任务
1. 理解docs_EN.md和alibi_cookbook.ipynb，并总结出alibi的解释器和解释方法
2. 了解geo.csv的数据结构，在接下来的脚本中，预测和模拟径流
3. 根据以上的解释器和方法，生成几个不同的py脚本并运行，得到解释性的图
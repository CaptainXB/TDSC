# xwz
# 2022/4/13 19:43
import xlrd
import matplotlib.pyplot as plt
from matplotlib import ticker
n = 10
data1_x = []
data1_y = []
data2_x = []
data2_y = []
for i in range(1, n+1):
    data1_x.append(i)
    data2_x.append(i)


oldWb = xlrd.open_workbook('D:\\实验数据\\iteration_new.xls')
newWS = oldWb.sheet_by_name('Sheet1')
for i in range(n):
    mm = 6
    data1_y.append(float(newWS.cell(214 + i, mm).value))
    data2_y.append(float(newWS.cell(320 + i, mm).value))

plt.plot(data1_x, data1_y, color='#3399FF', marker='o', alpha=1, linewidth=3, label='Bal', markersize=5)
plt.plot(data2_x, data2_y, color='#FF3B1D', marker='o', alpha=1, linewidth=3, label='AUC', markersize=5)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.03))
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=1)
plt.xlabel('T')
plt.ylabel('Prediction Results')
y = []
for i in range(10):
    y.append(0.4+0.06*i)
plt.yticks(y)
x = []
# for j in range(1, n+1):
#     x.append(j)
# plt.xticks(x)
# plt.ylim(0.6, 2.5)
plt.xlim(0, n+0.5)

plt.savefig('D:\\实验数据\\迭代图\\lmsvm_'+str(n)+'.png')
plt.show()

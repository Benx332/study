from sklearn import  linear_model
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False    #样本数据

Xi=[[6],[8],[9],[10],[12],]
Yi=[[40],[56],[69],[77],[96]]     #设置模型

model=linear_model.LinearRegression()

model.fit(Xi,Yi)
#将线性回归模型拟合到数据中

y_plot=model.predict(Xi)


print("y=",model.coef_[0],"x+",model.intercept_)

print(model.score(Xi,Yi))

plt.scatter(Xi,Yi,color="red",label="sample data",linewidth=2)
plt.plot(Xi,y_plot,color="green",label="regression data",linewidth=2)
plt.ylabel("价格/元")
plt.xlabel("蛋糕尺寸/英寸")
plt.legend(loc="lower right")
plt.show()
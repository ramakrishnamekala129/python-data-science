from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#height,weight, shoe size
X=[[181 ,80,44],[177,70,43],[160,60,38],
	[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],
	[159,95,45],[171,75,42],[181,85,43]]
Y=['male','female','female','male','male','female','female',
	'male','male','female','male']

clf=tree.DecisionTreeClassifier()
clf.fit(X,Y)

clf_svm=SVC()
clf_svm.fit(X,Y)

clf_preceptron=Perceptron()
clf_preceptron.fit(X,Y)

clf_Kneigh=KNeighborsClassifier()
clf_Kneigh.fit(X,Y)

C=[[190,65,39],[198,45,35],[175,85,45],[185,75,38]]
B=['male','female','male','female']

pred_Kneigh=clf_Kneigh.predict(C)
acc_Kneigh=accuracy_score(B,pred_Kneigh)*100
print("KNeighbors:{}".format(acc_Kneigh))

pred_svm=clf_svm.predict(C)
svm_acc=accuracy_score(B,pred_svm)*100
print("SVM:{}".format(svm_acc))

pred_preceptron=clf_preceptron.predict(C)
acc_preceptron=accuracy_score(B,pred_Kneigh)*100
print("Perceptron:{}".format(acc_preceptron))


prediction=clf.predict(C)
acc_tree=accuracy_score(B,prediction)*100
print("DecisionTreeClassifier:{}".format(acc_tree))

if acc_Kneigh > svm_acc and acc_Kneigh> acc_preceptron:
	if acc_Kneigh > acc_tree:
		print("KNeighborsClassifier is Best")
	elif acc_Kneigh == acc_tree:
		print("KNeighborsClassifier and DecisionTreeClassifier is Best")
	else:
		print("DecisionTreeClassifier is Best")
elif acc_preceptron > svm_acc and acc_preceptron> acc_Kneigh: 
	if acc_preceptron > acc_tree: 
		print("Perceptron is Best")
	elif acc_preceptron == acc_tree:
		print("Perceptron and DecisionTreeClassifier is Best")
	else:
		print("DecisionTreeClassifier is Best")
else :
	if svm_acc > acc_tree: 
		print("SVM is Best")
	elif svm_acc == acc_tree:
		print("SVM and DecisionTreeClassifier is Best")
	else:	
		print("DecisionTreeClassifier is Best")


	

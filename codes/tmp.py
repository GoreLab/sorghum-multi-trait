



6690    377.403489
6691    387.790510
6692    390.707971
6693    389.950112


plt.subplot(4, 2, 1)
plt.hist(y_pred_cv1['dbn_cv1_height_30'])
plt.title('30')

plt.subplot(4, 2, 2)
plt.hist(y_pred_cv1['dbn_cv1_height_45'])
plt.title('45')

plt.subplot(4, 2, 3)
plt.hist(y_pred_cv1['dbn_cv1_height_60'])
plt.title('60')

plt.subplot(4, 2, 4)
plt.hist(y_pred_cv1['dbn_cv1_height_75'])
plt.title('75')

plt.subplot(4, 2, 5)
plt.hist(y_pred_cv1['dbn_cv1_height_90'])
plt.title('90')

plt.subplot(4, 2, 6)
plt.hist(y_pred_cv1['dbn_cv1_height_105'])
plt.title('105')

plt.subplot(4, 2, 7)
plt.hist(y_pred_cv1['dbn_cv1_height_120'])
plt.title('120')

plt.show()
plt.clf()



plt.subplot(3, 2, 1)
plt.hist(y_pred_cv2['dbn_cv2_height_30~45'])
plt.title('45')

plt.subplot(3, 2, 2)
plt.hist(y_pred_cv2['dbn_cv2_height_30~60'])
plt.title('60')

plt.subplot(3, 2, 3)
plt.hist(y_pred_cv2['dbn_cv2_height_30~75'])
plt.title('75')

plt.subplot(3, 2, 4)
plt.hist(y_pred_cv2['dbn_cv2_height_30~90'])
plt.title('90')

plt.subplot(3, 2, 5)
plt.hist(y_pred_cv2['dbn_cv2_height_30~105'])
plt.title('105')

plt.show()
plt.clf()



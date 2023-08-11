# NGV_project
## 230811
- Data-driven Optimization (given dataset from LDW)
- Problems: outliers in the first QoI, named "CP_case0"
- Outliers make the training of GPR impossible (see the results below)
![image](https://github.com/sunwoong-yang/NGV_project/assets/65647892/1db489bd-4a6e-43b4-b7ed-c6c759393305)
- Therefore, outliers in the training dataset are removed manually as follows
  ```python 
  self.x_train = np.delete(self.x_train,[14,50],axis=0)
  self.y_train = np.delete(self.y_train,[14,50],axis=0)
  ```

## Todo (230816)
### 1. The way to get input files should be changed
  - Previous: whole dataset (initial + added pts) is input so that test dataset changes every iteration
    - Problem: test dataset used for plotting changes every iteration
  - Will be modified as below
    - Training dataset will not be input as a single excel file. Instead, excel file at each iteration will be input, and will be merged in the method "read_excel"
    - Test dataset will be fixed (not selected from the newly added queries, but only from the initial samples with fixed random seed)

### 2. Constraints will be handled
  - Previous: only the objective functions are viable
  - Will be modified to have constraints also

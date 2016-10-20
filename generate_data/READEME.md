# Generate Data
---
### sig_matching_data.m
Simulating the motion of the truck's changing lane, and saving the 8 channel sensor data in files.

### match_sigs_create_training_snippets.m
Loading the front radar data of the 4 trucks, generating 100 positive data and 100 negative data(Each negative data has 20 good points and 5 bad points followed), returning a dictionary contains these data.

### generate_data.py
Calling sig_matching_data.m to simulate 4 trucks' motion independently, then using matching_sigs_create_training_snippets.m to generate raw training data. 

### visualize_data.py
Providing a method that allows to visualize raw trainning data or scaled trainning data.

### scale_data.py
Providing a method to scale raw training data. It firstly shifts each data by its own mean and then scale the whole data with a common scaling factor.

### store_data.py
Storing the scaled training data in a pickle file 'front_dist_data'.

### *oct2py*
Containing the examples of using oct2py, which can run octave m-files from python. https://pypi.python.org/pypi/oct2py

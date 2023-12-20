""" 
This module is gonna be the center of control of my programm. That's where I am gonna lunch the execution of the different parts of the code. Depending on what I want to do I will run part(s) of the project or the whole one. For instance, if I want to test the complete programm I will be able to run all with one click. At the opposite, if I want to test a new module functionnning, I will run this one only. 

Modules as useful_functions and constant_variables are not imported here as already imported inside modules that use them.
"""


#src.data.make_dataset
if __name__ == '__main__':
    from data.make_dataset import *
def make_dataset_module_execution():
    """ 
        This function only executes the whole make_dataset module.
    """    
    # convert csv files into a big dataframe
    dataset = read_data(files)
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path of the dataset destination
    dataset_destination_path = os.path.join(current_dir, "..\\data\\interim\\data_interim.pkl")
    # Export the dataset to the path constructed
    dataset.to_pickle(dataset_destination_path)



#We import the dataset, it's useful for all the execution below
if __name__ == '__main__':
    dataset = load_data()



#src.vizualisation.visualize
if __name__ == '__main__':
    from visualization.visualize import *   
def visualize_module_execution():
    """ 
        This function only executes the whole visualize module.
        There are several functions to run in this module that are very specific. They are made to identify specific features outliers. We better have to go to the module to run it. That's reserved to particular uses in outliers identification.
    """   
    plot_all_num_features(dataset, save = False, density_estimate = False)
    

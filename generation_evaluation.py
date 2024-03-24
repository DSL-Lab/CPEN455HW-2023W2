'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify the remaining code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
2. Modify how you call your sample function(line 31)
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import torch
# You should modify this sample function to get the generated images from the model
# This function should save the generated images to the gen_data_dir, 
# which is fixed as 'samples/Class0', 'samples/Class1', 'samples/Class2', 'samples/Class3'
# Begin of your code
def sample():
    pass
# End of your code

if __name__ == "__main__":
    ref_data_dir = "data/test"
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gen_data_dir_list = ["samples/Class0", "samples/Class1", "samples/Class2", "samples/Class3"]
    fid_score_average = 0
    for gen_data_dir in gen_data_dir_list:
        if not os.path.exists(gen_data_dir):
            os.makedirs(gen_data_dir)
        #Begin of your code
        sample()
        #End of your code
        paths = [gen_data_dir, ref_data_dir]
        print("#generated images: {:d}, #reference images: {:d}".format(
            len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

        try:
            fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
            print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir_list))
        except:
            print("Dimension {:d} fails!".format(192))
            
        fid_score_average = fid_score_average + fid_score
        
    fid_score_average = fid_score_average / len(gen_data_dir_list)
    print("Average fid score: {}".format(fid_score_average))

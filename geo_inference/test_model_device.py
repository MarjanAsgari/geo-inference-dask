


import torch

if __name__ == "__main__":
   

    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(i)
    model1 = torch.jit.load("/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_model/4cls_RGB_5_1_2_3_scripted.pt",
        map_location=torch.device("cuda:1")
    )
    model2= model1.to("cuda:1")
    print(model1.device)
    print(model2.device)


